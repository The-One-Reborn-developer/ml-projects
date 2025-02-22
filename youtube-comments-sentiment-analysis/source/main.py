from kagglehub import dataset_download

from pathlib import Path

from fastai.text.all import (
    TextDataLoaders,
    text_classifier_learner,
    accuracy,
    AWD_LSTM
)


dataset_path_string = dataset_download('atifaliak/youtube-comments-dataset')
dataset_path = Path(dataset_path_string)

data_block = TextDataLoaders.from_csv(
    dataset_path,
    csv_fname='YoutubeCommentsDataSet.csv',
    text_col='Comment',
    label_col='Sentiment'
)


learner = text_classifier_learner(data_block, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
learner.fine_tune(4, 1e-2)

learner.predict('Whoever came up with this idea needs a huge raise. The last joker would have been very interesting to see too')
learner.predict('The man on the wheelchair has my respect 🗿💯')
learner.predict('Another murderer')
learner.predict('Ngl, that shit was SICK')
learner.predict("Why are blacks so obsessed with Aura? It feels animalistic to me, like how blacks are obsessed with rap and rapping about how great they are. It's just peacocking to attract a mate. I personally don't have a hyper strong desire to constantly have to look cool. I just want to chill and enjoy life. I get sick and tired of constantly dealing with mid looking people acting like they are gods gift to Earth.")
