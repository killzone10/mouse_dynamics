import numpy as np

# THIS FILE CONTAINS CONSTANTS #
ALL_USERS = [7, 9, 12, 15, 16, 20, 21, 23, 29, 35]
TEST_SIZE = 0.2
RANDOM_STATE = np.random.seed(0)
## FOLDER ## DATASET

BALABIT = 0
CHAOSHEN = 1
SINGAPUR = 2 



BASE_FOLDER = ['Mouse-Dynamics-Challenge-master-2']
TRAINING_FOLDER = ['training_files']
TEST_FOLDER   = ['test_files']
PUBLIC_LABELS = 'public_labels.csv'
TRAINING_FEATURE_FILENAME  = 'output/balabit_features_training.csv'
TEST_FEATURE_FILENAME  = 'output/balabit_features_test.csv'



## ACTIONS ##
# action codes: MM - mouse move; PC - point click; DD - drag and drop

MM = 1
PC = 3
DD = 4


## ACTION LENGHT ##
GLOBAL_MIN_ACTION_LENGTH = 4  ## MM ACTION CONSISTS OF MINIMUM lenght 4
CURV_THRESHOLD = 0.0005 # threshold for curvature
GLOBAL_MIN_TIME = 10


## HEADERS ##

ACTION_CSV_HEADER = ["type_of_action,traveled_distance_pixel,elapsed_time,direction_of_movement,straightness,num_points,sum_of_angles,mean_curv,sd_curv,max_curv,min_curv," \
                    "mean_omega,sd_omega,max_omega,min_omega,largest_deviation,dist_end_to_end_line,num_critical_points,"+\
                    "mean_vx,sd_vx,max_vx,min_vx,mean_vy,sd_vy,max_vy,min_vy,mean_v,sd_v,max_v,min_v,mean_a,sd_a,max_a,min_a,mean_jerk,sd_jerk,max_jerk,min_jerk,a_beg_time,userid"+\
                    "\n"]

ACTION_CSV_HEADER_LEGALITY = ["type_of_action,traveled_distance_pixel,elapsed_time,direction_of_movement,straightness,num_points,sum_of_angles,mean_curv,sd_curv,max_curv,min_curv," \
                    "mean_omega,sd_omega,max_omega,min_omega,largest_deviation,dist_end_to_end_line,num_critical_points,"+\
                    "mean_vx,sd_vx,max_vx,min_vx,mean_vy,sd_vy,max_vy,min_vy,mean_v,sd_v,max_v,min_v,mean_a,sd_a,max_a,min_a,mean_jerk,sd_jerk,max_jerk,min_jerk,a_beg_time,userid,legality"+\
                    "\n"]
ACTION_FILENAME = 'output/balabit_actions.csv'


## SCREEN LIMIT ##

X_THRESHOLD = 4000
Y_THRESHOLD = 4000