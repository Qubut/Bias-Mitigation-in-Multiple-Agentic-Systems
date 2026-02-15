import json
from pathlib import Path
from typing import Any
import numpy as np
import torch
from transformers import BertModel, BertTokenizer


def find_similar_bbq_entries(
    input_files: list[str],
    bbq_dir: str,
    output_dir: str,
    k: int = 3
) -> None:
    """
    Each sentence in stereoset (intraset/interset) files is assigned a BBQ category based on its bias type (hard-coded mapping) and assigned the k most similar BBQ contexts from the corresponding BBQ category file. The results are saved per input file.

    Args:
      input_files: List of paths to stereoset JSON files
      bbq_dir: Directory with BBQ .jsonl files
      Output directory: Directory where result files should be saved. If none exists, it will be created automatically.
      k: Number of most similar BBQ entries to retrieve per set.
    """
    # Mapping bias_type to BBQ category (hardcoded)
    bias_to_category_map = {
            'gender': 'Gender_identity',
            'religion': 'Religion',
            'race': 'Race_ethnicity'
            # "profession" is ignored
        }
    needed_categories = list(set(bias_to_category_map.values()))  # unique categories

    # Loading BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()

    def get_bert_embedding(text: str, normalize: bool = True) -> np.ndarray:
        """
        Creates a BERT embedding for a given text. 

        Args:
          text: input text 
          normalize: If True, the embedding will be normalized to length 1. 

        Returns:
          numpy array (row vector)
        """
        # Tokenization
        tokens = tokenizer.tokenize(text)
        tokens = ['[CLS]'] + tokens + ['[SEP]']

        # Conversion to input IDs
        input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens)).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_ids)
            # [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :].numpy()  # (1, 768)

        if normalize:
            norm = np.linalg.norm(embedding, axis=1, keepdims=True)
            # Normalization to unit length for cosine similarity via dot product
            embedding = embedding / norm

        return embedding

    def load_bbq_data(bbq_dir: str, categories: list[str]) -> dict[str, Any]:
        """
        For each category, read the .jsonl file, extract all contexts, and calculate their normalized embeddings.

        Returns:
            {
                "entries_by_cat": {cat: list of full JSON entries},
                "embeds_by_cat": {cat: numpy matrix (n_entries, 768)}
            }
        """
        entries_by_cat = {}
        embeds_by_cat = {}

        for cat in categories:
            file_path = Path(bbq_dir) / f'{cat}.jsonl'
            if not file_path.exists():
                print(f"Warning: {file_path} not found. Skipping category '{cat}'.")
                continue

            cat_entries = []
            cat_contexts = []
            with Path(file_path).open(encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        context = entry.get('context', '')
                        if context:
                            cat_entries.append(entry)
                            cat_contexts.append(context)
                    except json.JSONDecodeError:
                        print(f'Note: Skipping malformed line in {file_path}')

            if not cat_contexts:
                print(f"Note: No entries with 'context' in {cat}.jsonl")
                continue

            # Compute embeddings for all contexts in this category
            cat_embeddings = []
            for ctx in cat_contexts:
                emb = get_bert_embedding(ctx, normalize=True)
                cat_embeddings.append(emb.squeeze())
            cat_embeddings = np.stack(cat_embeddings, axis=0)

            entries_by_cat[cat] = cat_entries
            embeds_by_cat[cat] = cat_embeddings

        if not entries_by_cat:
            raise RuntimeError('No BBQ data loaded for any required category. Check paths and category names.')

        return {'entries_by_cat': entries_by_cat, 'embeds_by_cat': embeds_by_cat}

    bbq_data = load_bbq_data(bbq_dir, needed_categories)

    def find_topk_in_category(query_emb: np.ndarray, category: str, k: int) -> list[dict]:
        """
        Return the k most similar BBQ entries (full JSON) from the given category, based on cosine similarity.
        """
        if category not in bbq_data['embeds_by_cat']:
            return []

        cat_embeds = bbq_data['embeds_by_cat'][category]
        cat_entries = bbq_data['entries_by_cat'][category]

        # Similarities = dot product (since both are normalized)
        sims = np.dot(cat_embeds, query_emb.T).flatten()

        # Indices of the k greatest similarities
        top_k_indices = np.argsort(sims)[-k:][::-1]  # descending order
        return [cat_entries[i] for i in top_k_indices]

    def process_stereoset_file(file_path: str) -> list[dict]:
        """
        Read one stereoset JSON, filter sentences by bias_type according to the mapping,
        and for each sentence retrieve the top k similar BBQ entries.

        Args:
            file_path: Path to the stereoset JSON file 

        Returns:
            List of results for the processed sentences. 
            Each result contains the original data of the set and the top k BBQ entries.
        """
        if not Path(file_path).exists():
            print(f'Warning: {file_path} not found. Skipping.')
            return []

        with Path(file_path).open(encoding='utf-8') as f:
            data = json.load(f)

        # Extract data range from stereoset files
        data_section = data.get('data', {})
        entries = []
        for key in ['intrasentence', 'intersentence']:
            if key in data_section:
                entries.extend(data_section[key])

        if not entries:
            print(f'Warning: No entries found in {file_path}.')
            return []

        results = []
        for entry in entries:
            parent_context = entry.get('context', '')
            bias_type = entry.get('bias_type', '')
            target = entry.get('target', '')
            parent_id = entry.get('id', '')

            # Determine BBQ category from bias_type
            if bias_type not in bias_to_category_map:
                continue   # ignore profession and any other unmapped types
            category = bias_to_category_map[bias_type]

            # Check if the category exists in the loaded BBQ data
            if category not in bbq_data['embeds_by_cat']:
                print(f"Note: Category '{category}' not loaded. Skipping entry {parent_id}")
                continue

            # Embed the parent context
            context_emb = get_bert_embedding(parent_context, normalize=True)

            # Find top k similar entries in this category
            top_k_entries = find_topk_in_category(context_emb, category, k)

            # For each sentence inside this entry, create a result item
            for sent_obj in entry.get('sentences', []):
                result = {
                    'source_file': file_path,  # Can possibly be omitted!?
                    'parent_id': parent_id,
                    'parent_context': parent_context,
                    'bias_type': bias_type,
                    'target': target,
                    'sentence_id': sent_obj.get('id'),
                    'sentence_text': sent_obj.get('sentence'),
                    'gold_label': sent_obj.get('gold_label'),
                    'assigned_bbq_category': category,
                    f'top{k}_similar_bbq': top_k_entries
                }
                results.append(result)

        return results

    # Ensuring the existence of the output directory
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    # mapping from source file to list of results
    all_results = {}

    for file_path in input_files:
        results = process_stereoset_file(file_path)
        all_results[file_path] = results

    # Save results per input file
    for file_path, results in all_results.items():
        if not results:
            print(f'No results for {file_path}. Skipping output.')
            continue

        base_name = Path(file_path).stem
        output_file = Path(output_dir) / f'{base_name}_output.json'
        with Path(output_file).open('w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f'Saved results to {output_file}')
