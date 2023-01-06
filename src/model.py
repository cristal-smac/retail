import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import pandas as pd
import copy
import time

HISTORY_LENGTH = 50
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def H1(x, sigma=5, mu=10):
    if x < 10:
        return 4* (((-(x)**2) + (20)*(x))/20) -2
    else:
        return 8

def prospect(x):
    if x<=0:
        return (-np.exp(-x*1.1)+1)
    else:
        return (np.exp(x)-1)

def sat(x,space):
    c = C
    y = math.log(c-1)
    if x >=space:
        return  c / (1+math.exp(-((x-space)/SAT_ALPHA)+y))
    elif x >= -space:
        return 1
    else:
        return  c / (1+math.exp(-((x+space)/SAT_ALPHA)+y))
    
def reject_outliers(data, m = 300):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    if data[s<m] is None:
        return data
    else:
        return data[s<m]


def generate_price_sensibility():
    return {"inertie":np.random.uniform(1,2,1),"price":np.random.uniform(20,21,1),
            "quality":np.random.uniform(1,2,1),"promophile":np.random.uniform(1,2,1)}
            #,"jackpot":np.random.uniform(1,2,1),"advertising":np.random.uniform(1,2,1),"last_price":np.random.uniform(1,2,1)}

def generate_quality_sensibility():
    return {"inertie":np.random.uniform(1,2,1),"price":np.random.uniform(1,2,1),
            "quality":np.random.uniform(20,21,1),"promophile":np.random.uniform(1,2,1)}
            #,"jackpot":np.random.uniform(1,2,1),"advertising":np.random.uniform(1,2,1),"last_price":np.random.uniform(1,2,1)}

def generate_promophile_sensibility():
    return {"inertie":np.random.uniform(1,2,1),"price":np.random.uniform(1,2,1),
            "quality":np.random.uniform(1,2,1),"promophile":np.random.uniform(20,21,1)}
            #,"jackpot":np.random.uniform(1,2,1),"advertising":np.random.uniform(1,2,1),"last_price":np.random.uniform(1,2,1)}
def generate_inertie_sensibility():
    return {"inertie":np.random.uniform(20,21,1),"price":np.random.uniform(1,2,1),
            "quality":np.random.uniform(1,2,1),"promophile":np.random.uniform(1,2,1)}
            #,"jackpot":np.random.uniform(1,2,1),"advertising":np.random.uniform(1,2,1),"last_price":np.random.uniform(1,2,1)}

def generate_random_sensibility():
    return {"inertie":np.random.uniform(1,2,1),"price":np.random.uniform(1,2,1),
            "quality":np.random.uniform(1,2,1),"promophile":np.random.uniform(1,2,1)}
            #,"jackpot":np.random.uniform(1,2,1),"advertising":np.random.uniform(1,2,1),"last_price":np.random.uniform(1,2,1)}
    
def create_random_products_categorie(name):
    base_price = np.random.randint(10,100)
    products = []
    for i in range(6):
        products.append(Product((name + str(i)), base_price + np.random.randint(-base_price/2,base_price/2),np.random.random(), np.random.randint(1,10)))
    return ProductsCategorie(name, products)


def generate_product_categorie(name):
    base_price = np.random.randint(10, 100)
    products = []
    for j in range(3):
        price = base_price + np.random.normal(0,(base_price/4))
        quality = np.random.normal((price/100),0.1)
        for i in range(3): 
            products.append(Product((name + str(i+(j*3))), price, np.absolute(quality), 2 * (i+1)))
            price = price * 0.9
    return ProductsCategorie(name, products)

def generate_product_categorie_two(name):
    base_price = np.random.randint(10, 100)
    products = []
    for i in range(2): 
        price = base_price + np.random.normal(0,(base_price/4))
        quality = np.random.normal((price/100),0.1)
        products.append(Product((name + str(i)), price, np.absolute(quality), 2 * (i+1)))
        # price = price * 0.9
    return ProductsCategorie(name, products, np.random.random())

def generate_price_war_categorie(name):
    base_price = 10
    products = [Product("A", 10, 0.5, 1), Product("B", 12, 0.7, 1)]
    return ProductsCategorie(name, products)

def History_gen(categories,val_lambda):
    history = {}
    quantity_by_category = {}
    mini = val_lambda - 2.5
    for cat in categories:
        quantity_by_category[cat.name] = np.random.randint(mini, np.random.randint(mini+5,15), HISTORY_LENGTH)
        history[cat.name] = [np.random.choice(cat.product_list) for i in range(HISTORY_LENGTH)]
    return [quantity_by_category,history]
        

class Profile:
    def __init__(self, sensibility):
        assert len(sensibility)==4
        sum_s = sum(sensibility)
        self.price = sensibility[0]/sum_s
        self.quality = sensibility[1]/sum_s
        self.inertia = sensibility[2]/sum_s
        self.promo = sensibility[3]/sum_s
    def get_profile(self):
        return {"inertie":self.inertia,"price":self.price,
            "quality":self.quality,"promophile":self.promo}

class Agent:
    """
    This is the most complex class. It represent an agent with sensibility to price, quality etc... 
    and with needs for each categorie of product. At each time step, the agent "go to the store", 
    that mean for each categorie of product it compute a threshold using price, promotion, quality,
    inertie, advertising... and it's own "unique" sensibility. It then choose the 
    product which fit the most. Agents are initialized automaticaly throught the environement class. 
    """
    
    def __init__(self, name, env, history=None,profile=None):
        """
        """
        self.name = name
        # We assume that all agents have a fidelity card for now.
        self.jackpot = 0
        self.needs = {}
        self.type = np.random.choice([0, 1, 2, 3, 4])
        if profile == None:
            self.sensibility = [generate_price_sensibility(),generate_quality_sensibility(),
                                generate_promophile_sensibility(),generate_inertie_sensibility(),generate_random_sensibility()][self.type]
        else:
            self.sensibility = profile
#         self.sensibility = generate_price_sensibility()
        sum_norm = sum(self.sensibility.values())
        for sensi in self.sensibility.keys():
            self.sensibility[sensi] = self.sensibility[sensi]/sum_norm
        
        self.history_price_quality = {}
        self.inertie = {}
        self.ref = {}
        self.env = env
        self.history = {}
        self.attractivity_threshold = np.random.uniform(0,0.9) #The threshold used to check if the agent goes to this market or another
        self.quantity_by_category = {}
        self.max_quantity_by_category = {}
        self.track_bought_to_plot = {}
        # Besoin à calibrer, il faut l'associer aux achats moyen précédents ou relié au profil.
        for product_category in self.env.products_categories:
            if history is None:
                base_need = np.random.randint(1,10)
                mini = np.random.randint(0,5)
                self.quantity_by_category[product_category.name] = np.random.randint(mini, np.random.randint(mini+5,15), HISTORY_LENGTH) #List of lasts quantity bought
                # self.max_quantity_by_category[product_category.name] = np.mean(self.quantity_by_category[product_category.name]) * MAX_QUANTITY
                self.history[product_category.name] = [np.random.choice(product_category.product_list) for i in range(HISTORY_LENGTH)] #Liste et quantitée glissante. 
                self.needs[product_category.name] = np.mean(reject_outliers(self.quantity_by_category[product_category.name]))
                self.inertie[product_category.name] = [0, 0]
                self.track_bought_to_plot[product_category.name] = np.zeros(env.HP["NB_TICKS"], dtype=object)
            else:
                self.quantity_by_category = history[0]
                self.history = history[1]
                self.needs[product_category.name] = np.mean(self.quantity_by_category[product_category.name])
                self.inertie[product_category.name] = [0, 0]
                self.track_bought_to_plot[product_category.name] = np.zeros(env.HP["NB_TICKS"], dtype=object)
            self.history_price_quality[product_category.name] = []
            tmp_p = 0
            tmp_q = 0
            for tmp in self.history[product_category.name]:
                self.history_price_quality[product_category.name] += [(tmp.promotion_price,tmp.quality)]
                tmp_p += tmp.promotion_price
                tmp_q += tmp.quality
            tmp_p = tmp_p / HISTORY_LENGTH
            tmp_q = tmp_q / HISTORY_LENGTH
            self.ref[product_category.name] = Product("ref",tmp_p,tmp_q,1)
#             tmp_p = 0
#             tmp_q = 0
#             for product in self.history_price_quality[product_category.name]:
#                 tmp_p += product.price 
#                 tmp_q += product.quality
#             tmp_p = tmp_p / HISTORY_LENGTH
#             tmp_q = tmp_q / HISTORY_LENGTH
#             self.ref[product_category.name] = Product("ref",tmp_p,tmp_q,1)
    
    def get_product_inertia(self,product_category):
        dict_product_freq = self.compute_freq_products(product_category)
        return max(dict_product_freq)
    
    def get_sensibility(self):
        return self.sensibility
    
    def get_history(self):
        res = {}
        for i in self.history.keys():
            res [i] = []
            for j in range(len(self.history[i])):
                res [i] += [(self.history[i][j],self.quantity_by_category[i][j])]
        return res
    
    def go_store(self):
        """
        This is used to simulate an agent going to the supermarket.
        For each category of product the agent have to choose 1 product to buy.
        To do that it compute a threshold, and compare it's needs to this threshold.
        If it's needs >= threshold he consider this product. Then when th agent have 
        all the product it considers. It choose the product with the min threshold/one_pack_quantity.
        The one_pack_quantity coresponding to the number of product in the pack.
        """
        for product_categorie in self.env.products_categories:
            possible_buy = []
            dict_product_freq = self.compute_freq_products(product_categorie)
            l = len(self.quantity_by_category[product_categorie.name])
            last_buy = self.quantity_by_category[product_categorie.name][-1]
            if last_buy == 0:
                last_buy = 0.01
            p = sigmoid((self.needs[product_categorie.name]/last_buy))
            assert(p<=1 and p>=0), f"Variable p is : {p} and variable need is {self.needs}"
            # The agent is not interested in the category. 
            if np.random.random()>p:
                self.history[product_categorie.name] = np.append(self.history[product_categorie.name][1:],[None])
                self.quantity_by_category[product_categorie.name] = np.append(self.quantity_by_category[product_categorie.name][1:],[0]) 
                #Update the need (mean of quantity bought last HISTORY_LENGTH ticks)
                self.needs[product_categorie.name] = np.mean(reject_outliers(self.quantity_by_category[product_categorie.name]))
                self.env.non_buyers[self.env.tick] += 1
                if self.env.trace:
                    print("Pas de temps num : ",self.env.tick," Agent : ",self.name," Besoin", self.needs[product_categorie.name]," Achat : Rien", "Quantité : 0")
                return 0
            else: 
                for product in product_categorie.product_list:
                    # If the product is in state  a 3 buy 1 free for example.
                    if product.special_promo_product is not None:
                        utility, quantity = self.compute_utility(product.special_promo_product, product_categorie, dict_product_freq)
                        # We add the base product with the promoted product for the compute of the number of this kind of product bought 
                        # If product.one_pack_quantity (ex: un paquet de pates de 800g) < quantity (je veux acheter 1kg)
                        if product.special_promo_product.one_pack_quantity < quantity:
                            possible_buy.append([(product.special_promo_product, product), quantity, utility])
                    # We compute the utility of the product 
                    utility, quantity = self.compute_utility(product,
                                                             product_categorie,
                                                             dict_product_freq)
                    if product.one_pack_quantity <= quantity:
                        possible_buy.append([(product,
                                              product.special_promo_product),
                                             quantity, utility])
                if len(possible_buy) > 1:
                    dtype = object
                    possible_buy = np.array(possible_buy,dtype=object)
                    # The agent choose the most attractive for him.
                    somme = np.sum(possible_buy[:,2])
                    if somme == 0:
                        proba = None
                    else:
                        proba = []
                        for i in possible_buy[:, 2]:
                            proba += [i/somme]
                    chosen_product = possible_buy[np.random.choice(np.arange(0,len(possible_buy)),p=proba)]
                    self.buy(chosen_product[1],
                             product_categorie,
                             chosen_product[0])
                else:
                    if len(possible_buy) == 1:
                        chosen_product = possible_buy[0]
                        self.buy(chosen_product[1],
                                 product_categorie,
                                 chosen_product[0])
                    else:
                        self.history[product_categorie.name] = np.append(self.history[product_categorie.name][1:],[None])
                        self.quantity_by_category[product_categorie.name] = np.append(self.quantity_by_category[product_categorie.name][1:], [0])
                        # Append 0 a the end and delete first element
                        self.needs[product_categorie.name] = np.mean(reject_outliers(self.quantity_by_category[product_categorie.name]))
                        # Update the need (mean of quantity bought last H_len ticks)
                        self.env.non_buyers[self.env.tick] += 1
                        if self.env.trace:
                            print("Pas de temps num : ",self.env.tick," Agent : ",self.name," Besoin", self.needs[product_categorie.name]," Achat : Rien", "Quantité : 0")
                return 0

    def compute_quantity(self, product_category):
        moy = self.needs[product_category.name]
        data = self.quantity_by_category[product_category.name][-4:]
        quantity = max(0, moy+(np.sum(-data+moy)))
        q = int(np.random.normal(quantity, 0.5,1))
        if np.random.random() < quantity-q:
            q += 1
        return q
    
    def compute_freq_products(self, product_categorie):
        dict_product_freq = {}
        for p in self.history[product_categorie.name]:
            if not p is None:
                if p in dict_product_freq.keys():
                    dict_product_freq[p] += 1
                else:
                    dict_product_freq[p] = 1
        return dict_product_freq

    def compute_freq_products_lasts(self, product_categorie):
        dict_product_freq = {}
        for p in self.history[product_categorie.name][-10:]:
            if not p is None:
                if p in dict_product_freq.keys():
                    dict_product_freq[p] += 1
                else:
                    dict_product_freq[p] = 1
        return dict_product_freq

    def sat(self,U,space):
        x = (U)
        c = self.env.HP["C"]
        y = math.log(c-1)
        return c / (1 + math.exp(-x/self.env.HP["SAT_ALPHA"]+y))

    def compute_utility(self, product, product_categorie,
                        dict_product_freq, seconde_product=None):
        """
        For this agent, this function computes his opinion of the product in parameters.
        Using the sensibilities of the agent we compute a "threshold" (à changer de nom)
        using sum(sensibility_i * product_attribut_corresponding). 
        This function is principaly used to know which product is the most suitable for this agent
        comparing to the others product of the category this agent considers. 
        The agent choose considering product in the go_store decision function. 
        """
        # Check the most bought product. (Le produit de référence)
        # Il n'y a besoin de calculer ref qu'une seul fois,
        # a mettre dans une fonction seul.
        ref = self.ref[product_categorie.name]
        #max(dict_product_freq, key=dict_product_freq.get)
        # Le produit de référence est celui le plus acheté
        # dict_product_freq[max(dict_product_freq, key=dict_product_freq.get)] 
        # pour avoir la fréquence.
        dict_product_freq = self.compute_freq_products_lasts(product_categorie)
        if product in dict_product_freq.keys():
            inertie = H1(dict_product_freq[product])
        else:
            inertie = H1(0)
        # if np.random.random() < product.product_ad:
        #     touched_by_ad = 0
        # else:
        #     touched_by_ad = 1
        is_promo = 0
        if product.is_promo == 1:
            is_promo = 1
        threshold = ((self.sensibility["quality"]*(self.env.HP["QUALITY"]*(max(0, product.quality-ref.quality)+self.env.HP["PHI"]*(max(0, ref.quality-product.quality)))))+
                     (self.sensibility["price"]*(self.env.HP["PRICE"]*(self.env.HP["PHI"] *max(0, (product.promotion_price) -(ref.promotion_price)) +max(0, (ref.promotion_price) -(product.promotion_price))))) +
                     (self.sensibility["inertie"]*(inertie * self.env.HP["INERTIA"])) +
                     (self.sensibility["promophile"]*(self.env.HP["PROMO_U"] * is_promo)))
        if threshold<0:
            threshold = 0
        U = threshold
            
        quantity = self.compute_quantity(product_categorie) 
        if quantity < 0:
            quantity = 0
        quantity = np.random.normal(quantity * (self.sat(U,20)),quantity/4)
        if quantity < 0:
            quantity = 0
        threshold=float(threshold)
        return threshold, quantity
        
    def compute_utility_ref(self,product,dict_product_freq):
        ref = product
        U = ((self.sensibility["quality"] * (self.env.HP["QUALITY"]* (self.env.HP["PHI"]* max(0,product.quality - ref.quality)+ (max(0,ref.quality-product.quality))) ))+
            (self.sensibility["price"] * (self.env.HP["PRICE"]* (self.env.HP["PHI"] *max(0,(product.promotion_price) - (ref.promotion_price))+ max(0,(ref.promotion_price) - (product.promotion_price))))) )#+
#             (self.sensibility["last_price"] * (product.promotion_price/product.lasts_price[0]))+
            #(self.sensibility["inertie"] * (inertie * self.env.HP["INERTIA"]))+
#            (self.sensibility["advertising"] * touched_by_ad)+
            #(self.sensibility["promophile"] * (self.env.HP["PROMO_U"] * product.is_promo)))
        return U


    # TODO ajouter les promotion types carte fidélité 

    def buy(self,quantity,product_categorie,product):
        """
        Compute the number of pack the agent buy.
        Decrease the needs of agents, add revenues 
        and the numbuer of pruct bought in attributes of agent.
        """
        #If my need is less than the quantity in this pack of product  
#         if self.attractivity_threshold > self.env.attractivity:
#             return 0 #Nothing happend
        if product[1] is None:
            product_to_incr = product[0]
            product = product[0]
        else:
            product_to_incr = product[1]
            product = product[0]
        nb_pack_buy = int(quantity/product.one_pack_quantity)
        if np.random.random() < (quantity/product.one_pack_quantity - nb_pack_buy):
            nb_pack_buy += 1
        self.history_price_quality[product_categorie.name] = self.history_price_quality[product_categorie.name] [1:] + [(product.promotion_price,product.quality)]
        tmp_p = sum(i for i, j in self.history_price_quality[product_categorie.name]) / HISTORY_LENGTH
        tmp_q = sum(j for i, j in self.history_price_quality[product_categorie.name]) / HISTORY_LENGTH
        self.ref[product_categorie] = Product("ref",tmp_p,tmp_q,1)
        self.quantity_by_category[product_categorie.name]=np.append(self.quantity_by_category[product_categorie.name][1:],[nb_pack_buy * product.one_pack_quantity])
        self.needs[product_categorie.name] = np.mean(reject_outliers(self.quantity_by_category[product_categorie.name]))
        self.inertie[product_categorie.name][0] = product
        self.history[product_categorie.name] = np.append(self.history[product_categorie.name][1:],[product])
        product.nb_bought += (nb_pack_buy *product.one_pack_quantity) #1
        self.env.one_tick_revenues += nb_pack_buy * product.pack_price
        self.env.one_tick_sells_quantity += nb_pack_buy * product.one_pack_quantity
        self.track_bought_to_plot[product_categorie.name][self.env.tick] = product
        if self.env.trace:
            print("Pas de temps num : ",self.env.tick," Agent : ",self.name," Besoin", self.needs[product_categorie.name]," Achat : ", product.name, "Quantité : ",nb_pack_buy)
        return 0
    
class Environment:
    """
    This class needs products_categories class filled with products at the creation to work properly.
    creation: Environement([product_categorie1,product categorie2,...])
    When it's initialized, you just need to parametrize global variable NB_AGENTS , NB_TICKS and lauch the simulation with run()
    launch simulation: environment.run()
    The revenues over time are compute in the variable environment.revenues. It's a 1D array. 
    Plot revenues: plt.plot(environement.revenues)  with matplotlib.pyplot imported as plt

    """

    def __init__(self, products_categories, NB_AGENTS=50, NB_TICKS=52, agent_data=None):
        """
        Initialize the environement of the simulation.
        Categories of products and products whithin thoses categories are the environement and we create our agents.
        """
        self.tick = 0
        self.promo = [0] * NB_TICKS
        self.reduce = [0] * NB_TICKS
        self.HP = {}
        self.HP["NB_AGENTS"] = NB_AGENTS
        self.HP["NB_TICKS"] = NB_TICKS
        self.agents = []
        self.most_buy= {}
        self.products_categories = products_categories
        for category in products_categories:
            self.most_buy[category] = {}
            for p in category.product_list:
                self.most_buy[category][p.name] = []
        self.one_tick_revenues = 0
        self.one_tick_sells_quantity = 0
        self.quantity_sells = np.zeros(NB_TICKS+1)
        self.revenues = np.zeros(NB_TICKS)
        self.cumulative_nb_bought_per_product = {}
        self.non_buyers = []
        self.attractivity = 1
        self.attractivity_change = -0.002
        for i in range(NB_AGENTS):
            if agent_data is None:
                self.agents.append(Agent("agent " + str(len(self.agents)),self))
            else:
                self.agents.append(Agent("agent " + str(len(self.agents)),self, agent_data[i]))
    def add_agent(self,agents_to_add):
        for agent in agents_to_add:
            self.agents.append(agent)

    def initialise(self, price=0.5, quality=100, promophile=5, sat_alpha=100, phi=0.3, c = 2, inertia = 2):
        self.tick = 0
        self.most_buy= {}
        for category in self.products_categories:
            self.most_buy[category] = {}
            for p in category.product_list:
                self.most_buy[category][p.name] = []
        self.one_tick_revenues = 0
        self.one_tick_sells_quantity = 0
        self.quantity_sells = np.zeros(self.HP["NB_TICKS"]+1)
        self.revenues = np.zeros(self.HP["NB_TICKS"])
        self.cumulative_nb_bought_per_product = {}
        self.non_buyers = []
#         self.HP["PROMO"] = promotion
        self.HP["PRICE"] = price
        self.HP["QUALITY"] = quality
        self.HP["PROMO_U"] = promophile
        self.HP["SAT_ALPHA"] = sat_alpha
        self.HP["PHI"] = -1 - phi
#         self.HP["gamma"] = arg_gamma
        self.HP["C"] = c
        self.HP["INERTIA"] = inertia
        return 0
    """
    The main function of the simulation.
    Compute the number of time steps and call function to call agents.
    """

    def run(self,trace=False):
        self.trace = trace
        for categories in self.products_categories:
            for product in categories.product_list:
                self.cumulative_nb_bought_per_product[product.name] = [0]
        cpt = 0
        for i in range(self.HP["NB_TICKS"]):
            # print("TICK",i)
            self.non_buyers += [0]
            self.run_once()  # The main call is here.
            for categories in self.products_categories:
                self.count_most_buy(categories)
                for product in categories.product_list:
                    product.refresh_lasts_prices
                    self.cumulative_nb_bought_per_product[product.name] += [product.nb_bought]#/(i+1)]
                    product.nb_bought = 0
            self.tick += 1
            self.revenues[i] = self.one_tick_revenues
            self.quantity_sells[i] = self.one_tick_sells_quantity
            self.one_tick_revenues = 0
            self.one_tick_sells_quantity=0
            
            #Regler pb promo sur le même tick
            tmp_promo = self.promo[i]
            if tmp_promo != 0:
                if tmp_promo[0] == 0:
                    self.products_categories[tmp_promo[1]].product_list[tmp_promo[2]].simple_promotion(tmp_promo[3])
                else:
                    if tmp_promo[3] == 0:
                        self.products_categories[tmp_promo[1]].product_list[tmp_promo[2]].unmake_pack_promotion()
                    else:
                        self.products_categories[tmp_promo[1]].product_list[tmp_promo[2]].make_pack_promotion(tmp_promo[3][0],tmp_promo[3][1])
            tmp_reduce = self.reduce[i]
            if tmp_reduce != 0:
                self.products_categories[tmp_reduce[1]].product_list[tmp_reduce[2]].reduce_price(tmp_reduce[0])
    
    def make_promo(self,promo_type):
        if promo_type[0] == 0:
            self.promo[promo_type[2][0]] = (0,promo_type[3],promo_type[4],promo_type[1])
            self.promo[promo_type[2][1]] = (0,promo_type[3],promo_type[4],0)
        else:
            self.promo[promo_type[2][0]] = (1,promo_type[3],promo_type[4],promo_type[1])
            self.promo[promo_type[2][0]] = (1,promo_type[3],promo_type[4],0)
            
    def reduce_price(self,percent, tick, cat ,prod):
        self.reduce[tick] = (percent,cat,prod)
    
    def count_most_buy(self,category):
        # if not self.most_buy[category]:
        #     self.most_buy[category] = {}
        res = {}
        for p in category.product_list:
            res[p.name] = 0
        for agent in self.agents:
            d = agent.compute_freq_products(category)
            res[max(d,key=d.get).name] += 1
            # for k in d.keys():
            #     res[k.name] += d[k]
        for p in category.product_list:
            self.most_buy[category][p.name] += [res[p.name]]
        return 0

    def changing_attractivity(self):
        """
        At each time step the attractivity of the supermarket/store change, if there is promotions,
        it raise, if there is no promtion it goes down. 
        """
        self.attractivity += self.attractivity_change

    def run_once(self):
        """
        This function call all agents one by one so they chose and buy products.
        """
        for agent in self.agents:
            agent.go_store()
        self.changing_attractivity()

    def get_history_CA(self):
        res = np.zeros(52)
        for nb_agent in range(self.HP["NB_AGENTS"]):
            for cpt in range(len(self.agents[nb_agent].history["bananes"])):
                res[cpt] += self.agents[nb_agent].quantity_by_category["bananes"][cpt] * self.agents[nb_agent].history["bananes"][cpt].price
        return res

    def get_history_nb_buy(self):
        res = np.zeros(52)
        for cat in self.products_categories:
            for nb_agent in range(self.HP["NB_AGENTS"]):
                for cpt in range(len(self.agents[nb_agent].history[cat.name])):
                    res[cpt] += self.agents[nb_agent].quantity_by_category[cat.name][cpt]# * environement.agents[nb_agent].history["bananes"][cpt].price
        return res
    
    def show_sales_separated(self):
        categorie = self.products_categories[0]
        cpt = 0
        fig, axs = plt.subplots(len(categorie.product_list), 1, sharex=True,figsize=(16,9))
        for product in categorie.product_list:
            axs[cpt].plot(self.cumulative_nb_bought_per_product[product.name])
            axs[cpt].legend([product.name,])
#             axs[cpt].set_ylim(-5, self.max(self.cumulative_nb_bought_per_product[product.name]))
            cpt+=1
        plt.show()
        return 0
    
    def show_sales(self):
        categorie = self.products_categories[0]
        for product in categorie.product_list:
            plt.plot(self.cumulative_nb_bought_per_product[product.name])
        plt.show()
        return 0
    
    def get_sales(self):
        categorie = self.products_categories[0]
        res = []
        for product in categorie.product_list:
            res += [self.cumulative_nb_bought_per_product[product.name]]
        return np.array(res).T
    
    def get_favorites(self):
        res = {}
        for cat in self.most_buy.keys():
            for product_name in self.most_buy[cat].keys():
                #plt.plot(environement.most_buy[cat][product_name])
                res[product_name] = self.most_buy[cat][product_name]
        return res
    
class ProductsCategorie:
    """
    Is a categorie of product, containing a list a products of the same sort like bananas, or water. 
    """

    def __init__(self, name, product_list):
        """
        A category of product is composed of it's product list of this category, the name of the category and the mean price of this category.
        """
        self.name = name
        self.product_list = product_list
        # self.mean_need_growth = mean_need_growth #Alpha
#         self.mean_price = 0
#         for i in product_list:
#             # We compute the mean price of the categorie to use it to check if a product is cheaper than the other product or more expensive.
#             self.mean_price += (i.price/i.one_pack_quantity)
#         self.mean_price = self.mean_price/len(product_list)

    def add_product(self,one_product):
        """
        one_product: type Product. A product to add at this category of product. 
        This just add a new product into this category. TODO find a way to avoid multiple same product.
        """
        self.product_list.insert(0,one_product)
        return 0
        
    def get_products(self):
        return self.product_list
    
    def show_products(self):
        for product in self.product_list:
            print("Nom : ",product.get_name(), ", Prix total : ",product.get_price_total())
        return 0


class Product:
    """
    This is a product, it can be anything, it has a name, a price, a quality between 0 and 1 and the quantity of product in one pack. 
    For exemple if I want to represent a pack of 6 bottle of water, one_pack_quantity = 6.
    """

    def __init__(self, name, price, quality, one_pack_quantity):
        """
        Init of all the attributes
        name : str, the name of the product
        is_promo : bool, 0 the product is not in promotion, 1 is in promotion
        promotion_price : float, The price of 1 pack, it can change over time, it's the equivalent of the price displayed in the market.
        lasts_price : list, The price of the product on the 3 last time step 
        price : float, The base price of 1 pack. The more one_pack_quantity is high the more the price will be high too.
        quality: float, between 0 and 1 the quality of a product
        product_ad: float, between 0 and 1, it's the advertising on this particular product.
        nb_bought: int, The number of this product bought during simulation, updated in the buy function of agents.
        special_promo_product, list, This parameter is used to setup 3 buy 1 free promotions.
        """
        self.name = name
        self.is_promo = 0
        self.promotion_price = price
        self.lasts_price = [price,price,price] #On enregsitre les 3 denières valeurs.
        self.price = price #Price 1 pack (price for 1 unit * one_pack_quantity)
        self.quality = quality
        self.percent_packpot = 0 #How much % is rewarded on the jackpot of the consumer. To calibrate with real data.
        self.one_pack_quantity = one_pack_quantity
        self.pack_price = one_pack_quantity * price
        self.product_ad = 0 # Between 0 and 1, 1 everybody is touched by the ad, 0 nobody. 0.5 mean 50%
        self.nb_bought = 0
        self.special_promo_product = None 
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name

    def simple_promotion(self, percentage):
        """
        percentage : int, between 0 and 100
        Update the promotion price and the attribute is_promo to
        correspond to a perentage% promotion. product.simple_promotion(30)
        will make a 30% promotion on this product.
        """
        self.promotion_price = self.price * (1 - (percentage/100))
        self.pack_price = self.promotion_price * self.one_pack_quantity
        self.lasts_price[2] = self.lasts_price[1]
        self.lasts_price[1] = self.lasts_price[0]
        self.lasts_price[0] = self.promotion_price
        self.is_promo = 1
        if percentage == 0:
            self.is_promo = 0
            
    def reduce_price(self, percentage):
        """
        percentage : int, between 0 and 100
        Update the promotion price and the attribute is_promo to
        correspond to a perentage% promotion. product.simple_promotion(30)
        will make a 30% promotion on this product.
        """
        promo = self.price / self.promotion_price
        self.price = self.price * (1 - (percentage/100))
        self.promotion_price = self.price / promo
        self.pack_price = self.promotion_price * self.one_pack_quantity
        self.lasts_price[2] = self.lasts_price[1]
        self.lasts_price[1] = self.lasts_price[0]
        self.lasts_price[0] = self.promotion_price
        
    def raise_price(self, percentage):
        """
        percentage : int, between 0 and 100
        Update the promotion price and the attribute is_promo to
        correspond to a perentage% promotion. product.simple_promotion(30)
        will make a 30% promotion on this product.
        """
        promo = self.price / self.promotion_price
        self.price = self.price * (1 - (percentage/100))
        self.promotion_price = self.price / promo
        self.pack_price = self.promotion_price * self.one_pack_quantity
        self.lasts_price[2] = self.lasts_price[1]
        self.lasts_price[1] = self.lasts_price[0]
        self.lasts_price[0] = self.promotion_price

    def refresh_lasts_prices(self):
        """
        Update the value of lasts_prices, it is supposed to be used 
        at each time step on each agents.
        """
        self.lasts_price[2] = self.lasts_price[1]
        self.lasts_price[1] = self.lasts_price[0]
        self.lasts_price[0] = self.promotion_price

    def make_pack_promotion(self, nb_to_buy, nb_free):  # Implémentation d'une promotion 2 acheté 1 gratuit serait, pack_promotion(2,1)
        """
        This function make a special promotion. If an agent buy nb_to_buy 
        + nb_free it costs him only nb_to_buy.
        """
        self.special_promo_product = Product(self.name+" special promo",
                                             self.price * (nb_to_buy/(nb_to_buy+nb_free)),
                                             self.quality,
                                             self.one_pack_quantity*
                                             (nb_to_buy+nb_free))
        self.special_promo_product.is_promo = 1
        self.is_promo = 1

    def unmake_pack_promotion(self):
        """
        Unmake the promotion of certain amount buy => certain amount free. (x acheté y offert)
        """
        self.special_promo_product = None
        self.is_promo = 0
        
    def get_price_unit(self):
        return self.price
    
    def get_price_total(self):
        return (self.price*self.one_pack_quantity)
    
    def get_name(self):
        return self.name
    

    
