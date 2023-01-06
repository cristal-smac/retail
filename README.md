# tools for multiagent retail simulations

This repository contain a simulator of price dynamics and is made to contain tools related to simulation in marketing dynamics and simulation using ABM and individual centered approach.

## Price dynamics, simulate discounts in a store

### Quick start
	
	from src.model import *
	import matplotlib.pyplot as plt
	cat_0 = ProductsCategorie("catégorie 0",[Product("Produit_A", 10, 0.5, 1), Product("Produit_B", 12, 0.7, 1)])
	env = Environment([cat_0],300,10)
	env.initialise()
	env.run()
	results = env.get_favorites()
	plt.plot(results)