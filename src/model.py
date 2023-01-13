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

# def sat(x,space):
#     c = C
#     y = math.log(c-1)
#     if x >=space:
#         return  c / (1+math.exp(-((x-space)/SAT_ALPHA)+y))
#     elif x >= -space:
#         return 1
#     else:
#         return  c / (1+math.exp(-((x+space)/SAT_ALPHA)+y))
    
def reject_outliers(data, m = 300):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    if data[s<m] is None:
        return data
    else:
        return data[s<m]


def generate_price_sensibility():
    return {"inertie":np.random.uniform(1,2,1),"price":np.random.uniform(10,21,1),
            "quality":np.random.uniform(1,2,1),"promophile":np.random.uniform(1,2,1)}
            #,"jackpot":np.random.uniform(1,2,1),"advertising":np.random.uniform(1,2,1),"last_price":np.random.uniform(1,2,1)}

def generate_quality_sensibility():
    return {"inertie":np.random.uniform(1,2,1),"price":np.random.uniform(1,2,1),
            "quality":np.random.uniform(10,21,1),"promophile":np.random.uniform(1,2,1)}
            #,"jackpot":np.random.uniform(1,2,1),"advertising":np.random.uniform(1,2,1),"last_price":np.random.uniform(1,2,1)}

def generate_promophile_sensibility():
    return {"inertie":np.random.uniform(1,2,1),"price":np.random.uniform(1,2,1),
            "quality":np.random.uniform(1,2,1),"promophile":np.random.uniform(10,21,1)}
            #,"jackpot":np.random.uniform(1,2,1),"advertising":np.random.uniform(1,2,1),"last_price":np.random.uniform(1,2,1)}
def generate_inertie_sensibility():
    return {"inertie":np.random.uniform(10,21,1),"price":np.random.uniform(1,2,1),
            "quality":np.random.uniform(1,2,1),"promophile":np.random.uniform(1,2,1)}
            #,"jackpot":np.random.uniform(1,2,1),"advertising":np.random.uniform(1,2,1),"last_price":np.random.uniform(1,2,1)}

def generate_random_sensibility():
    return {"inertie":np.random.uniform(1,2,1),"price":np.random.uniform(1,2,1),
            "quality":np.random.uniform(1,2,1),"promophile":np.random.uniform(1,2,1)}
            #,"jackpot":np.random.uniform(1,2,1),"advertising":np.random.uniform(1,2,1),"last_price":np.random.uniform(1,2,1)}
    
def create_random_packs_categorie(name):
    base_price = np.random.randint(10,100)
    packs = []
    for i in range(6):
        packs.append(pack((name + str(i)), base_price + np.random.randint(-base_price/2,base_price/2),np.random.random(), np.random.randint(1,10)))
    return packsCategorie(name, packs)


def generate_pack_categorie(name):
    base_price = np.random.randint(10, 100)
    packs = []
    for j in range(3):
        price = base_price + np.random.normal(0,(base_price/4))
        quality = np.random.normal((price/100),0.1)
        for i in range(3): 
            packs.append(pack((name + str(i+(j*3))), price, np.absolute(quality), 2 * (i+1)))
            price = price * 0.9
    return packsCategorie(name, packs)

def generate_pack_categorie_two(name):
    base_price = np.random.randint(10, 100)
    packs = []
    for i in range(2): 
        price = base_price + np.random.normal(0,(base_price/4))
        quality = np.random.normal((price/100),0.1)
        packs.append(pack((name + str(i)), price, np.absolute(quality), 2 * (i+1)))
        # price = price * 0.9
    return packsCategorie(name, packs, np.random.random())

def generate_price_war_categorie(name):
    base_price = 10
    packs = [pack("A", 10, 0.5, 1), pack("B", 12, 0.7, 1)]
    return packsCategorie(name, packs)

def History_gen(categories,val_lambda,hist_l):
    history = {}
    quantity_by_category = {}
    mini = val_lambda - 2.5
    for cat in categories:
        quantity_by_category[cat.name] = np.random.randint(mini, np.random.randint(mini+5,15), hist_l)
        history[cat.name] = [np.random.choice(cat.pack_list) for i in range(hist_l)]
    return [quantity_by_category,history]
        

class Profil:
    def __init__(self, sensibility):
        assert len(sensibility.keys())==4
        sum_s = sum(sensibility.values())
        self.price = sensibility["price"]/sum_s
        self.quality = sensibility["quality"]/sum_s
        self.inertia = sensibility["inertia"]/sum_s
        self.promo = sensibility["promophile"]/sum_s
    def get_profil(self):
        return {"inertie":self.inertia,"price":self.price,
            "quality":self.quality,"promophile":self.promo}

class Agent:
    """
    This is the most complex class. It represent an agent with sensibility to price, quality etc... 
    and with needs for each categorie of pack. At each time step, the agent "go to the store", 
    that mean for each categorie of pack it compute a threshold using price, promotion, quality,
    inertie, advertising... and it's own "unique" sensibility. It then choose the 
    pack which fit the most. Agents are initialized automaticaly throught the environement class. 
    """
    
    def __init__(self, name, env, history=None,profil=None):
        """
        """
        self.name = name
        # We assume that all agents have a fidelity card for now.
        self.jackpot = 0
        self.needs = {}
        self.type = np.random.choice([0, 1, 2, 3, 4])
        if profil == None:
            self.sensibility = [generate_price_sensibility(),generate_quality_sensibility(),
                                generate_promophile_sensibility(),generate_inertie_sensibility(),generate_random_sensibility()][self.type]
        else:
            self.sensibility = profil.get_profil()
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
        for pack_category in self.env.packs_categories:
            if history is None:
                base_need = np.random.randint(1,10)
                mini = np.random.randint(0,5)
                self.quantity_by_category[pack_category.name] = np.random.randint(mini, np.random.randint(mini+5,15), env.HP["HIST_L"]) #List of lasts quantity bought
                # self.max_quantity_by_category[pack_category.name] = np.mean(self.quantity_by_category[pack_category.name]) * MAX_QUANTITY
                self.history[pack_category.name] = [np.random.choice(pack_category.pack_list) for i in range(env.HP["HIST_L"])] #Liste et quantitée glissante. 
                self.needs[pack_category.name] = np.mean(reject_outliers(self.quantity_by_category[pack_category.name]))
                self.inertie[pack_category.name] = [0, 0]
                self.track_bought_to_plot[pack_category.name] = np.zeros(env.HP["NB_TICKS"], dtype=object)
            else:
                assert(len(history[1][pack_category.name])==(env.HP["HIST_L"]))#f"Variable History length is : {len(history[1]} and the history length of ABM need is {env.HP["HIST_L"]}"
                self.quantity_by_category = history[0]
                self.history = history[1]
                self.needs[pack_category.name] = np.mean(self.quantity_by_category[pack_category.name])
                self.inertie[pack_category.name] = [0, 0]
                self.track_bought_to_plot[pack_category.name] = np.zeros(env.HP["NB_TICKS"], dtype=object)
            self.history_price_quality[pack_category.name] = []
            tmp_p = 0
            tmp_q = 0
            for tmp in self.history[pack_category.name]:
                self.history_price_quality[pack_category.name] += [(tmp.promotion_price,tmp.quality)]
                tmp_p += tmp.promotion_price
                tmp_q += tmp.quality
            tmp_p = tmp_p / env.HP["HIST_L"]
            tmp_q = tmp_q / env.HP["HIST_L"]
            self.ref[pack_category.name] = Pack("ref",tmp_p,tmp_q,1)
#             tmp_p = 0
#             tmp_q = 0
#             for pack in self.history_price_quality[pack_category.name]:
#                 tmp_p += pack.price 
#                 tmp_q += pack.quality
#             tmp_p = tmp_p / HISTORY_LENGTH
#             tmp_q = tmp_q / HISTORY_LENGTH
#             self.ref[pack_category.name] = pack("ref",tmp_p,tmp_q,1)
    
    def __get_pack_inertia(self,pack_category):
        dict_pack_freq = self.compute_freq_packs(pack_category)
        return max(dict_pack_freq)
    
    def get_profil(self):
        return self.sensibility
    
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
        For each category of pack the agent have to choose 1 pack to buy.
        To do that it compute a threshold, and compare it's needs to this threshold.
        If it's needs >= threshold he consider this pack. Then when th agent have 
        all the pack it considers. It choose the pack with the min threshold/one_pack_quantity.
        The one_pack_quantity coresponding to the number of pack in the pack.
        """
        for pack_categorie in self.env.packs_categories:
            possible_buy = []
            dict_pack_freq = self.compute_freq_packs(pack_categorie)
            l = len(self.quantity_by_category[pack_categorie.name])
            last_buy = self.quantity_by_category[pack_categorie.name][-1]
            if last_buy == 0:
                last_buy = 0.01
            p = sigmoid((self.needs[pack_categorie.name]/last_buy))
            assert(p<=1 and p>=0), f"Variable p is : {p} and variable need is {self.needs}"
            # The agent is not interested in the category. 
            if np.random.random()>p:
                self.history[pack_categorie.name] = np.append(self.history[pack_categorie.name][1:],[None])
                self.quantity_by_category[pack_categorie.name] = np.append(self.quantity_by_category[pack_categorie.name][1:],[0]) 
                #Update the need (mean of quantity bought last HISTORY_LENGTH ticks)
                self.needs[pack_categorie.name] = np.mean(reject_outliers(self.quantity_by_category[pack_categorie.name]))
                self.env.non_buyers[self.env.tick] += 1
                if self.env.trace:
                    print("Pas de temps num : ",self.env.tick," Agent : ",self.name," Besoin", self.needs[pack_categorie.name]," Achat : Rien", "Quantité : 0")
                return 0
            else: 
                for pack in pack_categorie.pack_list:
                    # If the pack is in state  a 3 buy 1 free for example.
                    if pack.special_promo_pack is not None:
                        utility, quantity = self.__compute_utility(pack.special_promo_pack, pack_categorie, dict_pack_freq)
                        # We add the base pack with the promoted pack for the compute of the number of this kind of pack bought 
                        # If pack.one_pack_quantity (ex: un paquet de pates de 800g) < quantity (je veux acheter 1kg)
                        if pack.special_promo_pack.one_pack_quantity < quantity:
                            possible_buy.append([(pack.special_promo_pack, pack), quantity, utility])
                    # We compute the utility of the pack 
                    utility, quantity = self.__compute_utility(pack,
                                                             pack_categorie,
                                                             dict_pack_freq)
                    if pack.one_pack_quantity <= quantity:
                        possible_buy.append([(pack,
                                              pack.special_promo_pack),
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
                    chosen_pack = possible_buy[np.random.choice(np.arange(0,len(possible_buy)),p=proba)]
                    self.buy(chosen_pack[1],
                             pack_categorie,
                             chosen_pack[0])
                else:
                    if len(possible_buy) == 1:
                        chosen_pack = possible_buy[0]
                        self.buy(chosen_pack[1],
                                 pack_categorie,
                                 chosen_pack[0])
                    else:
                        self.history[pack_categorie.name] = np.append(self.history[pack_categorie.name][1:],[None])
                        self.quantity_by_category[pack_categorie.name] = np.append(self.quantity_by_category[pack_categorie.name][1:], [0])
                        # Append 0 a the end and delete first element
                        self.needs[pack_categorie.name] = np.mean(reject_outliers(self.quantity_by_category[pack_categorie.name]))
                        # Update the need (mean of quantity bought last H_len ticks)
                        self.env.non_buyers[self.env.tick] += 1
                        if self.env.trace:
                            print("Pas de temps num : ",self.env.tick," Agent : ",self.name," Besoin", self.needs[pack_categorie.name]," Achat : Rien", "Quantité : 0")
                return 0

    def __compute_quantity(self, pack_category):
        moy = self.needs[pack_category.name]
        data = self.quantity_by_category[pack_category.name][-4:]
        quantity = max(0, moy+(np.sum(-data+moy)))
        q = int(np.random.normal(quantity, 0.5,1))
        if np.random.random() < quantity-q:
            q += 1
        return q
    
    def compute_freq_packs(self, pack_categorie):
        dict_pack_freq = {}
        for p in self.history[pack_categorie.name]:
            if not p is None:
                if p in dict_pack_freq.keys():
                    dict_pack_freq[p] += 1
                else:
                    dict_pack_freq[p] = 1
        return dict_pack_freq

    def compute_freq_packs_lasts(self, pack_categorie):
        dict_pack_freq = {}
        for p in self.history[pack_categorie.name][-10:]:
            if not p is None:
                if p in dict_pack_freq.keys():
                    dict_pack_freq[p] += 1
                else:
                    dict_pack_freq[p] = 1
        return dict_pack_freq

    def sat(self,U,space):
        x = (U)
        c = self.env.HP["C"]
        y = math.log(c-1)
        return c / (1 + math.exp(-x/self.env.HP["SAT_ALPHA"]+y))

    def __compute_utility(self, pack, pack_categorie,
                        dict_pack_freq, seconde_pack=None):
        """
        For this agent, this function computes his opinion of the pack in parameters.
        Using the sensibilities of the agent we compute a "threshold" (à changer de nom)
        using sum(sensibility_i * pack_attribut_corresponding). 
        This function is principaly used to know which pack is the most suitable for this agent
        comparing to the others pack of the category this agent considers. 
        The agent choose considering pack in the go_store decision function. 
        """
        # Check the most bought pack. (Le produit de référence)
        # Il n'y a besoin de calculer ref qu'une seul fois,
        # a mettre dans une fonction seul.
        ref = self.ref[pack_categorie.name]
        #max(dict_pack_freq, key=dict_pack_freq.get)
        # Le produit de référence est celui le plus acheté
        # dict_pack_freq[max(dict_pack_freq, key=dict_pack_freq.get)] 
        # pour avoir la fréquence.
        dict_pack_freq = self.compute_freq_packs_lasts(pack_categorie)
        if pack in dict_pack_freq.keys():
            inertie = H1(dict_pack_freq[pack])
        else:
            inertie = H1(0)
        # if np.random.random() < pack.pack_ad:
        #     touched_by_ad = 0
        # else:
        #     touched_by_ad = 1
        is_promo = 0
        if pack.is_promo == 1:
            is_promo = 1
        threshold = ((self.sensibility["quality"]*(self.env.HP["QUALITY"]*(max(0, pack.quality-ref.quality)+self.env.HP["PHI"]*(max(0, ref.quality-pack.quality)))))+
                     (self.sensibility["price"]*(self.env.HP["PRICE"]*(self.env.HP["PHI"] *max(0, (pack.promotion_price) -(ref.promotion_price)) +max(0, (ref.promotion_price) -(pack.promotion_price))))) +
                     (self.sensibility["inertie"]*(inertie * self.env.HP["INERTIA"])) +
                     (self.sensibility["promophile"]*(self.env.HP["PROMO_U"] * is_promo)))
        if threshold<0:
            threshold = 0
        U = threshold
            
        quantity = self.__compute_quantity(pack_categorie) 
        if quantity < 0:
            quantity = 0
        quantity = np.random.normal(quantity * (self.sat(U,20)),quantity/4)
        if quantity < 0:
            quantity = 0
        threshold=float(threshold)
        return threshold, quantity
        
    def ____compute_utility_ref(self,pack,dict_pack_freq):
        ref = pack
        U = ((self.sensibility["quality"] * (self.env.HP["QUALITY"]* (self.env.HP["PHI"]* max(0,pack.quality - ref.quality)+ (max(0,ref.quality-pack.quality))) ))+
            (self.sensibility["price"] * (self.env.HP["PRICE"]* (self.env.HP["PHI"] *max(0,(pack.promotion_price) - (ref.promotion_price))+ max(0,(ref.promotion_price) - (pack.promotion_price))))) )#+
#             (self.sensibility["last_price"] * (pack.promotion_price/pack.lasts_price[0]))+
            #(self.sensibility["inertie"] * (inertie * self.env.HP["INERTIA"]))+
#            (self.sensibility["advertising"] * touched_by_ad)+
            #(self.sensibility["promophile"] * (self.env.HP["PROMO_U"] * pack.is_promo)))
        return U


    # TODO ajouter les promotion types carte fidélité 

    def buy(self,quantity,pack_categorie,pack):
        """
        Compute the number of pack the agent buy.
        Decrease the needs of agents, add revenues 
        and the numbuer of pruct bought in attributes of agent.
        """
        #If my need is less than the quantity in this pack of pack  
#         if self.attractivity_threshold > self.env.attractivity:
#             return 0 #Nothing happend
        if pack[1] is None:
            pack_to_incr = pack[0]
            pack = pack[0]
        else:
            pack_to_incr = pack[1]
            pack = pack[0]
        nb_pack_buy = int(quantity/pack.one_pack_quantity)
        if np.random.random() < (quantity/pack.one_pack_quantity - nb_pack_buy):
            nb_pack_buy += 1
        self.history_price_quality[pack_categorie.name] = self.history_price_quality[pack_categorie.name] [1:] + [(pack.promotion_price,pack.quality)]
        tmp_p = sum(i for i, j in self.history_price_quality[pack_categorie.name]) / self.env.HP["HIST_L"]
        tmp_q = sum(j for i, j in self.history_price_quality[pack_categorie.name]) / self.env.HP["HIST_L"]
        self.ref[pack_categorie.name] = Pack("ref",tmp_p,tmp_q,1)
        self.quantity_by_category[pack_categorie.name]=np.append(self.quantity_by_category[pack_categorie.name][1:],[nb_pack_buy * pack.one_pack_quantity])
        self.needs[pack_categorie.name] = np.mean(reject_outliers(self.quantity_by_category[pack_categorie.name]))
        self.inertie[pack_categorie.name][0] = pack
        self.history[pack_categorie.name] = np.append(self.history[pack_categorie.name][1:],[pack])
        pack.nb_bought += (nb_pack_buy *pack.one_pack_quantity) #1
        self.env.one_tick_revenues += nb_pack_buy * pack.pack_price
        self.env.one_tick_sells_quantity += nb_pack_buy * pack.one_pack_quantity
        self.track_bought_to_plot[pack_categorie.name][self.env.tick] = pack
        if self.env.trace:
            print("Pas de temps num : ",self.env.tick," Agent : ",self.name," Besoin", self.needs[pack_categorie.name]," Achat : ", pack.name, "Quantité : ",nb_pack_buy)
        return 0
            
    
    
class SMA:
    """
    This class needs packs_categories class filled with packs at the creation to work properly.
    creation: Environement([pack_categorie1,pack categorie2,...])
    When it's initialized, you just need to parametrize global variable NB_AGENTS , NB_TICKS and lauch the simulation with run()
    launch simulation: environment.run()
    The revenues over time are compute in the variable environment.revenues. It's a 1D array. 
    Plot revenues: plt.plot(environement.revenues)  with matplotlib.pyplot imported as plt

    """

    def __init__(self, packs_categories, NB_AGENTS, NB_TICKS, h_length=50, agent_data=None):
        """
        Initialize the environement of the simulation.
        Categories of packs and packs whithin thoses categories are the environement and we create our agents.
        """
        self.tick = 0
        self.promo = [0] * NB_TICKS
        self.reduce = [0] * NB_TICKS
        self.HP = {}
        self.HP["NB_AGENTS"] = NB_AGENTS
        self.HP["NB_TICKS"] = NB_TICKS
        self.HP["HIST_L"] = h_length
        self.agents = []
        self.most_buy= {}
        self.packs_categories = packs_categories
        for category in packs_categories:
            self.most_buy[category] = {}
            for p in category.pack_list:
                self.most_buy[category][p.name] = []
        self.one_tick_revenues = 0
        self.one_tick_sells_quantity = 0
        self.quantity_sells = np.zeros(NB_TICKS+1)
        self.revenues = np.zeros(NB_TICKS)
        self.cumulative_nb_bought_per_pack = {}
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
        for category in self.packs_categories:
            self.most_buy[category] = {}
            for p in category.pack_list:
                self.most_buy[category][p.name] = []
        self.one_tick_revenues = 0
        self.one_tick_sells_quantity = 0
        self.quantity_sells = np.zeros(self.HP["NB_TICKS"]+1)
        self.revenues = np.zeros(self.HP["NB_TICKS"])
        self.cumulative_nb_bought_per_pack = {}
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
        for categories in self.packs_categories:
            for pack in categories.pack_list:
                self.cumulative_nb_bought_per_pack[pack.name] = [0]
        cpt = 0
        for i in range(self.HP["NB_TICKS"]):
            # print("TICK",i)
            self.non_buyers += [0]
            self.run_once()  # The main call is here.
            for categories in self.packs_categories:
                self.count_most_buy(categories)
                for pack in categories.pack_list:
                    pack.refresh_lasts_prices
                    self.cumulative_nb_bought_per_pack[pack.name] += [pack.nb_bought]#/(i+1)]
                    pack.nb_bought = 0
            self.tick += 1
            self.revenues[i] = self.one_tick_revenues
            self.quantity_sells[i] = self.one_tick_sells_quantity
            self.one_tick_revenues = 0
            self.one_tick_sells_quantity=0
            
            #Regler pb promo sur le même tick
            tmp_promo = self.promo[i]
            if tmp_promo != 0:
                if tmp_promo[0] == 0:
                    self.packs_categories[tmp_promo[1]].pack_list[tmp_promo[2]].simple_promotion(tmp_promo[3])
                else:
                    if tmp_promo[3] == 0:
                        self.packs_categories[tmp_promo[1]].pack_list[tmp_promo[2]].unmake_pack_promotion()
                    else:
                        self.packs_categories[tmp_promo[1]].pack_list[tmp_promo[2]].make_pack_promotion(tmp_promo[3][0],tmp_promo[3][1])
            tmp_reduce = self.reduce[i]
            if tmp_reduce != 0:
                self.packs_categories[tmp_reduce[1]].pack_list[tmp_reduce[2]].reduce_price(tmp_reduce[0])
    
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
        for p in category.pack_list:
            res[p.name] = 0
        for agent in self.agents:
            d = agent.compute_freq_packs(category)
            res[max(d,key=d.get).name] += 1
            # for k in d.keys():
            #     res[k.name] += d[k]
        for p in category.pack_list:
            self.most_buy[category][p.name] += [res[p.name]]
        return 0

    def __changing_attractivity(self):
        """
        At each time step the attractivity of the supermarket/store change, if there is promotions,
        it raise, if there is no promtion it goes down. 
        """
        self.attractivity += self.attractivity_change

    def run_once(self):
        """
        This function call all agents one by one so they chose and buy packs.
        """
        for agent in self.agents:
            agent.go_store()
        self.__changing_attractivity()

    def get_history_CA(self):
        res = np.zeros(52)
        for nb_agent in range(self.HP["NB_AGENTS"]):
            for cpt in range(len(self.agents[nb_agent].history["bananes"])):
                res[cpt] += self.agents[nb_agent].quantity_by_category["bananes"][cpt] * self.agents[nb_agent].history["bananes"][cpt].price
        return res

    def get_history_nb_buy(self):
        res = np.zeros(52)
        for cat in self.packs_categories:
            for nb_agent in range(self.HP["NB_AGENTS"]):
                for cpt in range(len(self.agents[nb_agent].history[cat.name])):
                    res[cpt] += self.agents[nb_agent].quantity_by_category[cat.name][cpt]# * environement.agents[nb_agent].history["bananes"][cpt].price
        return res
    
    def show_sales_separated(self):
        categorie = self.packs_categories[0]
        cpt = 0
        fig, axs = plt.subplots(len(categorie.pack_list), 1, sharex=True,figsize=(16,9))
        for pack in categorie.pack_list:
            axs[cpt].plot(self.cumulative_nb_bought_per_pack[pack.name])
            axs[cpt].legend([pack.name,])
#             axs[cpt].set_ylim(-5, self.max(self.cumulative_nb_bought_per_pack[pack.name]))
            cpt+=1
        plt.show()
        return 0
    
    def show_sales(self):
        categorie = self.packs_categories[0]
        for pack in categorie.pack_list:
            plt.plot(self.cumulative_nb_bought_per_pack[pack.name])
        plt.show()
        return 0
    
    def get_sales(self):
        categorie = self.packs_categories[0]
        res = []
        for pack in categorie.pack_list:
            res += [self.cumulative_nb_bought_per_pack[pack.name]]
        return np.array(res).T
    
    def get_favorites(self):
        res = {}
        for cat in self.most_buy.keys():
            for pack_name in self.most_buy[cat].keys():
                #plt.plot(environement.most_buy[cat][pack_name])
                res[pack_name] = self.most_buy[cat][pack_name]
        return res
    
    def get_turnover(self):
        return self.revenues
    
class Category:
    """
    Is a categorie of pack, containing a list a packs of the same sort like bananas, or water. 
    """

    def __init__(self, name, pack_list):
        """
        A category of pack is composed of it's pack list of this category, the name of the category and the mean price of this category.
        """
        self.name = name
        self.pack_list = pack_list
        # self.mean_need_growth = mean_need_growth #Alpha
#         self.mean_price = 0
#         for i in pack_list:
#             # We compute the mean price of the categorie to use it to check if a pack is cheaper than the other pack or more expensive.
#             self.mean_price += (i.price/i.one_pack_quantity)
#         self.mean_price = self.mean_price/len(pack_list)

    def add_pack(self,one_pack):
        """
        one_pack: type pack. A pack to add at this category of pack. 
        This just add a new pack into this category. TODO find a way to avoid multiple same pack.
        """
        self.pack_list.insert(0,one_pack)
        return 0
        
    def get_packs(self):
        return self.pack_list
    
    def show_packs(self):
        for pack in self.pack_list:
            print("Nom : ",pack.get_name(), ", Prix total : ",pack.get_price_total())
        return 0


class Pack:
    """
    This is a pack, it can be anything, it has a name, a price, a quality between 0 and 1 and the quantity of pack in one pack. 
    For exemple if I want to represent a pack of 6 bottle of water, one_pack_quantity = 6.
    """

    def __init__(self, name, price, quality, one_pack_quantity):
        """
        Init of all the attributes
        name : str, the name of the pack
        is_promo : bool, 0 the pack is not in promotion, 1 is in promotion
        promotion_price : float, The price of 1 pack, it can change over time, it's the equivalent of the price displayed in the market.
        lasts_price : list, The price of the pack on the 3 last time step 
        price : float, The base price of 1 pack. The more one_pack_quantity is high the more the price will be high too.
        quality: float, between 0 and 1 the quality of a pack
        pack_ad: float, between 0 and 1, it's the advertising on this particular pack.
        nb_bought: int, The number of this pack bought during simulation, updated in the buy function of agents.
        special_promo_pack, list, This parameter is used to setup 3 buy 1 free promotions.
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
        self.pack_ad = 0 # Between 0 and 1, 1 everybody is touched by the ad, 0 nobody. 0.5 mean 50%
        self.nb_bought = 0
        self.special_promo_pack = None 
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name

    def simple_promotion(self, percentage):
        """
        percentage : int, between 0 and 100
        Update the promotion price and the attribute is_promo to
        correspond to a perentage% promotion. pack.simple_promotion(30)
        will make a 30% promotion on this pack.
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
        correspond to a perentage% promotion. pack.simple_promotion(30)
        will make a 30% promotion on this pack.
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
        correspond to a perentage% promotion. pack.simple_promotion(30)
        will make a 30% promotion on this pack.
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
        self.special_promo_pack = pack(self.name+" special promo",
                                             self.price * (nb_to_buy/(nb_to_buy+nb_free)),
                                             self.quality,
                                             self.one_pack_quantity*
                                             (nb_to_buy+nb_free))
        self.special_promo_pack.is_promo = 1
        self.is_promo = 1

    def unmake_pack_promotion(self):
        """
        Unmake the promotion of certain amount buy => certain amount free. (x acheté y offert)
        """
        self.special_promo_pack = None
        self.is_promo = 0
        
    def get_price_unit(self):
        return self.price
    
    def get_price_total(self):
        return (self.price*self.one_pack_quantity)
    
    def get_name(self):
        return self.name
    

    
