# Configuration file for the Sphinx documentation builder.
# Kronos documentation — Anomalib-style with sphinx-design, grids, cards, tabs, dropdowns.

import sys
from pathlib import Path

# Define paths
project_root = Path(__file__).resolve().parent.parent
packages_path = project_root / 'packages'

# Add workspace packages so autodoc can import bias_mitigation.
sys.path.insert(0, str(project_root / 'src'))

project = 'Bias Mitigation in MAS'
copyright = 'Abdullah Alomar'  # noqa: A001
author = 'Abdullah Alomar'
release = '1.0.0'
version = '1.0.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'myst_parser',
    'sphinx_design',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosectionlabel',
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',
    'sphinxcontrib.mermaid',
]

# MyST configuration — full extension set like Anomalib
myst_enable_extensions = [
    'colon_fence',
    'linkify',
    'substitution',
    'tasklist',
    'deflist',
    'fieldlist',
]
myst_fence_as_directive = ['mermaid']
myst_heading_anchors = 3

# Autodoc configuration
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'show-inheritance': True,
    'undoc-members': True,
}
autodoc_typehints = 'description'
napoleon_use_param = True
suppress_warnings = [
    'sphinx_autodoc_typehints.forward_reference',
    'sphinx_autodoc_typehints.guarded_import',
]

# Enable section anchors for cross-referencing
autosectionlabel_prefix_document = True

# Copybutton configuration
copybutton_exclude = '.linenos, .gp, .go'

# Templates and patterns
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_css_files = ['css/custom.css']
html_title = 'Bias Mitigation Documentation'
html_theme_options = {
    'logo': {
        'text': 'Bias Mitigation in MAS',
    },
    'repository_url': 'https://github.com/Qubut/Bias-Mitigation-in-Multiple-Agentic-Systems',
    'use_repository_button': True,
    'show_toc_level': 2,
}

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'pydantic': ('https://docs.pydantic.dev/latest/', None),
}
