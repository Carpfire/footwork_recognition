from os.path import join

# Paths for sports video data
SPORTS_ROOT_DIR = 'data/sports' #'/mnt/nvme/sports'

FS_ROOT_DIR = join(SPORTS_ROOT_DIR, 'fs')
FS_POSE_DIR = join(FS_ROOT_DIR, 'pose')
FS_VIDEO_DIR = join(FS_ROOT_DIR, 'videos')
FS_CROP_DIR = join(FS_ROOT_DIR, 'crops')

FX_ROOT_DIR = join(SPORTS_ROOT_DIR, 'fx')
FX_POSE_DIR = join(FX_ROOT_DIR, 'pose')
FX_VIDEO_DIR = join(FX_ROOT_DIR, 'videos')
FX_CROP_DIR = join(FX_ROOT_DIR, 'crops')

DIVING48_ROOT_DIR = join(SPORTS_ROOT_DIR, 'diving48')
DIVING48_POSE_DIR = join(DIVING48_ROOT_DIR, 'pose')
DIVING48_VIDEO_DIR = join(DIVING48_ROOT_DIR, 'videos')
DIVING48_CROP_DIR = join(DIVING48_ROOT_DIR, 'crops')

TENNIS_ROOT_DIR = join(SPORTS_ROOT_DIR, 'tennis')
TENNIS_POSE_DIR = join(TENNIS_ROOT_DIR, 'pose')
TENNIS_VIDEO_DIR = join(TENNIS_ROOT_DIR, 'videos')
TENNIS_CROP_DIR = join(TENNIS_ROOT_DIR, 'player-crops')

FENCING_ROOT_DIR = 'C:\\Users\\liamc\\Desktop\\fencing_vision\\data'
FENCING_POSE_DIR = join(FENCING_ROOT_DIR, 'embeddings', '2d_pose')
FENCING_VIDEO_DIR = join(FENCING_ROOT_DIR, 'clips')
FENCING_CROP_DIR = join(FENCING_ROOT_DIR, 'crops')
