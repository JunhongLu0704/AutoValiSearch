# Example Experiment Trace

```text
LLM suggestion
  -> config generation
  -> controller validation
  -> aggregate execution over 4 splits x 2 seeds
  -> metric collection
  -> validation policy analysis
  -> next-round feedback
```

## Stage I Trace

The LLM proposes a configuration inside the bounded search space:

```json
{"lr": 0.01, "lambdap": 1.0, "epochp": 5, "num_f": 3}
```

The controller validates the schema, checks legal values, rejects duplicates, and dispatches the aggregate trial.

## Stage II Trace

The Val Designer proposes a checkpoint-selection policy. The controller compiles it, evaluates required validation views, and applies safety fallback rules before reporting deployable metrics.
