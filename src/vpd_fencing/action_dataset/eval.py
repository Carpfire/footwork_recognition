# Train and test split

# Hold out all videos from 2018
FS_TEST_PREFIXES = (
    # 'men_olympic_short_program_2010',
    # 'men_olympic_short_program_2014',
    'men_olympic_short_program_2018',
    # 'men_world_short_program_2017',
    'men_world_short_program_2018',
    # 'men_world_short_program_2019',
    # 'women_olympic_short_program_2014',
    'women_olympic_short_program_2018',
    # 'women_world_short_program_2017',
    'women_world_short_program_2018',
    # 'women_world_short_program_2019',
)

FENCING_TEST_PREFIXES = ('BLUE_ALEXANIN_KAZ_vs_BOREL_FRA')

# Hold out by match video
TENNIS_TEST_PREFIXES = (
    'usopen_2015_mens_final_federer_djokovic',
    # 'usopen_2019_mens_final_medvedev_nadal',
    # 'usopen_2019_womens_final_andreescu_williams',
    'usopen_2019_womens_osaka_gauff',
    # 'wimbledon_2018_mens_semifinal_djokovic_nadal',
    # 'wimbledon_2018_womens_quarterfinal_williams_giorgi',
    # 'wimbledon_2019_mens_final_djokovic_federer',
    'wimbledon_2019_mens_semifinal_federer_nadal',
    'wimbledon_2019_womens_final_halep_williams'
)
fencing_val_path = 'C:\\Users\\liamc\\Desktop\\fencing_vision\\vpd_fencing\\action_dataset\\fencing\\val.ids.txt'
# def _get_fencing_prefixes(path):
#     with open(fencing_val_path, 'r') as f:
#         prefixes = tuple(line.split(':', 1)[0] for line in f.readlines())
#     return prefixes

def _get_tennis_prefixes(video_list):
    return tuple('{}{}'.format(x, y) for x in ['', 'front__', 'back__']
                 for y in video_list)


def get_test_prefixes(dataset):
    if dataset.startswith('fs'):
        return FS_TEST_PREFIXES
    elif dataset.startswith('tennis'):
        return _get_tennis_prefixes(TENNIS_TEST_PREFIXES)
    elif dataset.startswith('fencing'):
        return FENCING_TEST_PREFIXES
    else:
        raise NotImplementedError('Unknown dataset:' + dataset)
