from transformers import pipeline


# Wrong 3/5


classifier = pipeline('sentiment-analysis')
print(classifier('Whoever came up with this idea needs a huge raise. The last joker would have been very interesting to see too'))
print(classifier('The man on the wheelchair has my respect 🗿💯'))
print(classifier('Another murderer'))
print(classifier('Ngl, that shit was SICK'))
print(classifier("Why are blacks so obsessed with Aura? It feels animalistic to me, like how blacks are obsessed with rap and rapping about how great they are. It's just peacocking to attract a mate. I personally don't have a hyper strong desire to constantly have to look cool. I just want to chill and enjoy life. I get sick and tired of constantly dealing with mid looking people acting like they are gods gift to Earth."))
