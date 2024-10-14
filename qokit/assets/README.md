The parameters should be loaded and saved with `orient='index'`:

```python
import pandas as pd
df = pd.read_json("best_LABS_QAOA_parameters_wrt_overlap.json", orient='index')
df.to_json('best_LABS_QAOA_parameters_wrt_overlap.json', orient='index', indent=4)
```
