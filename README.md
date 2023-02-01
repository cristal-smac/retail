# tools for multiagent retail simulations

This repository contain a simulator of price dynamics and is made to contain tools related to simulation in marketing dynamics and simulation using ABM and individual centered approach.

Team : CRISTAL Lab, SMAC team, Lille University

Company : fifty-five

Contact : jarod.vanderlynden@fifty-five.com, philippe.mathieu@univ-lille.fr, romain@fifty-five.com

## Price dynamics, simulate discounts in a store

### Quick start
	
```python
from src.model import *
import matplotlib.pyplot as plt
cat_0 = ProductsCategorie("catégorie 0",[Product("Produit_A", 10, 0.5, 1), Product("Produit_B", 12, 0.7, 1)])
sma = SMA([cat_0],300,10)
sma.initialise()
sma.run()
results = env.get_favorites()
plt.plot(results)
```
### Get started

A presentation of the model, all the posible methods and how to use it. This notebook is made to understand parameters of the model and its agents and how to use them properly. You can launch this notebook directly with the following link :

FR : [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cristal-smac/retail.git/master?filepath=FR_GetStarted.ipynb) EN : [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cristal-smac/retail.git/master?filepath=ENG_GetStarted.ipynb)


