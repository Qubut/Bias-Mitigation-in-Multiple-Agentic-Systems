```mermaid
stateDiagram-v2
    [*] --> genesis
    genesis --> interaction : step · cond has_peer_answers
    genesis --> genesis : step · cond !has_peer_answers
    genesis --> completed : finish
    interaction --> interaction : step · cond has_peer_answers
    interaction --> completed : finish
    completed --> [*]
```
