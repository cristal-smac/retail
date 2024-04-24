# tools for multiagent retail simulations

This repository contain a simulator of price dynamics and is made to contain tools related to simulation in marketing dynamics and simulation using ABM and individual centered approach.

Team : CRISTAL Lab, SMAC team, Lille University

Company : fifty-five

Contact : jarod.vanderlynden@fifty-five.com, philippe.mathieu@univ-lille.fr, romain@fifty-five.com

***

## Price dynamics, simulate discounts in a store

This section is dedicated to the model presented in the article "_Simulation of consumers behavior facing discounts and promotions_", presented in ICAART, and "_Understanding the Impact of Promotions on Consumer Behavior_" extended abstract AAMAS, in french "_Comprendre l'impact des stratégies de prix sur le comportement des consommateurs_" JFSMA, extended version "_Impact des stratégies de prix sur les consommateurs_" in ROIA.

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

"_GetStarted_" notebook is a presentation of the model, all the posible methods and how to use it. This notebook is made to understand parameters of the model and agents and how to use those parameters properly. You can launch this notebook directly with the following link :

FR : [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cristal-smac/retail.git/main?filepath=SupermarketModel/FR_GetStarted.ipynb) EN : [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cristal-smac/retail.git/main?filepath=SupermarketModel/ENG_GetStarted.ipynb)

### Experiments

"_Experiments_" notebook contains the various simulations presented in the article, with a fixed random seed to reproduce the same results. You can launch this notebook directly with the following link :

FR : [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cristal-smac/retail.git/main?filepath=SupermarketModel/FR_Experiments.ipynb) EN : [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cristal-smac/retail.git/master?filepath=SupermarketModel/ENG_Experiments.ipynb)

***

