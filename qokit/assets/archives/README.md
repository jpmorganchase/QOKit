### Contents

`best_known_QAOA_parameters_wrt_MF.json` : best known (with respect to merit factor) QAOA parameters for LABS problem. 
`best_known_QAOA_parameters_wrt_overlap.json` : best known (with respect to overlap) QAOA parameters for LABS problem. 

To load a dataframe containing both MF and overlap results:
```python
from utils import get_all_best_known
df = get_all_best_known()
```

To load just one dataframe:

```python
import pandas as pd
df = pd.read_json('assets/best_known_QAOA_parameters_wrt_MF.json', orient='index')
``` 

To add new parameters, load the dataframe as above and update the relevant row. Then save using:

```python
df.to_json('assets/best_known_QAOA_parameters_wrt_MF.json', indent=4, orient='index')
```

Don't forget to drop the irrelevant columns, otherwise errors may occur. Example (dropping terms):

```python
df.drop(['terms'], axis=1)
```


`mean_params_0227.json` and `median_params_0227.json` : Mean and median computed over optimized QAOA parameters for N >= 22