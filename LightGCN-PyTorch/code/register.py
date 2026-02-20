import world
import dataloader
import model
import utils
from pprint import pprint

if world.DATA_FORMAT == 'recdata':
    dataset = dataloader.RecDataLoader(
        config=world.config,
        data_root=world.REC_DATA_PATH,
        dataset=world.dataset,
    )
else:
    if world.dataset in ['gowalla', 'yelp2018', 'amazon-book']:
        dataset = dataloader.Loader(path="../data/"+world.dataset)
    elif world.dataset == 'lastfm':
        dataset = dataloader.LastFM()

print('===========config================')
pprint(world.config)
print("data_format:", world.DATA_FORMAT)
if world.DATA_FORMAT == 'recdata':
    print("recdata_root:", world.REC_DATA_PATH)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')

MODELS = {
    'mf': model.PureMF,
    'lgn': model.LightGCN
}
