```mermaid
stateDiagram-v2
    [*] --> genesis
    genesis --> interaction : advance
    interaction --> interaction : advance · cond !rounds_exhausted
    interaction --> completed : advance · cond rounds_exhausted
    completed --> [*]
```
