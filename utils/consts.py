import numpy as np

# THIS FILE CONTAINS CONSTANTS #
BALABIT_USERS = [7, 9, 12, 15, 16, 20, 21, 23, 29, 35]
CHAOSHEN_USERS = [i for i in range (1,29)]
SINGAPUR_USERS = [i for i in range (0,24)]

TEST_SIZE = 0.2
VALIDATION_SIZE = 0.5

NUM_ACTIONS = 1

RANDOM_STATE = np.random.seed(0)
## FOLDER ## DATASET

BALABIT = 0
CHAOSHEN = 1
SINGAPUR = 2 
DFL = 3


BASE_FOLDER = ['Mouse-Dynamics-Challenge-master-2', 'ChaoShenCSV', 'TWOS-Dataset','dfl']
TRAINING_FOLDER = ['training_files', '', 'mouse_ano','']
TEST_FOLDER   = ['test_files']
PUBLIC_LABELS = 'public_labels.csv'
STOLEN_SESSIONS_FILEPATH  = 'TWOS-Dataset\\ImportantInfo.xlsx'



## ACTIONS ##
# action codes: MM - mouse move; PC - point click; DD - drag and drop

MM = 1
PC = 3
DD = 4


## ACTION LENGHT ##
GLOBAL_MIN_ACTION_LENGTH = 4  ## MM ACTION CONSISTS OF MINIMUM lenght 4
CURV_THRESHOLD = 0.0005 # threshold for curvature
GLOBAL_MIN_TIME = 0.5 ## TODO

GLOBAL_MIN_TIME_DFL = 10000 ## ITS IN MILISECONDS SO SHOULD BE AT LEAST 300 500

GLOBAL_MIN_TIME_SINGAPUR = 2


GLOBAL_MIN_TIME_CHAOSHEN = 6000 ## TODO 
GLOBAL_MIN_ACTION_LENGHT_CHAOSHEN = 5 ## TODO

## HEADERS ##

BALABIT_ACTION = "type_of_action,traveled_distance_pixel,elapsed_time,direction_of_movement,straightness,num_points,sum_of_angles,mean_curv,sd_curv,max_curv,min_curv," \
                    "mean_omega,sd_omega,max_omega,min_omega,largest_deviation,dist_end_to_end_line,num_critical_points,"+\
                    "mean_vx,sd_vx,max_vx,min_vx,mean_vy,sd_vy,max_vy,min_vy,mean_v,sd_v,max_v,min_v,mean_a,sd_a,max_a,min_a,mean_jerk,sd_jerk,max_jerk,min_jerk,a_beg_time,userid"+\
                    "\n"

CHAOSHEN_ACTION = "type_of_action,traveled_distance_pixel,elapsed_time,direction_of_movement,straightness,num_points,sum_of_angles,mean_curv,sd_curv,max_curv,min_curv," \
                    "mean_omega,sd_omega,max_omega,min_omega,largest_deviation,dist_end_to_end_line,num_critical_points,"+\
                    "mean_vx,sd_vx,max_vx,min_vx,mean_vy,sd_vy,max_vy,min_vy,mean_v,sd_v,max_v,min_v,mean_a,sd_a,max_a,min_a,mean_jerk,sd_jerk,max_jerk,min_jerk,a_beg_time,userid"+\
                    "\n"
TWOS_ACTION = "type_of_action,traveled_distance_pixel,elapsed_time,direction_of_movement,straightness,num_points,sum_of_angles,mean_curv,sd_curv,max_curv,min_curv," \
                    "mean_omega,sd_omega,max_omega,min_omega,largest_deviation,dist_end_to_end_line,num_critical_points,"+\
                    "mean_vx,sd_vx,max_vx,min_vx,mean_vy,sd_vy,max_vy,min_vy,mean_v,sd_v,max_v,min_v,mean_a,sd_a,max_a,min_a,mean_jerk,sd_jerk,max_jerk,min_jerk,a_beg_time,userid"+\
                    "\n"

DFL_ACTION = "type_of_action,traveled_distance_pixel,elapsed_time,direction_of_movement,straightness,num_points,sum_of_angles,mean_curv,sd_curv,max_curv,min_curv," \
                    "mean_omega,sd_omega,max_omega,min_omega,largest_deviation,dist_end_to_end_line,num_critical_points,"+\
                    "mean_vx,sd_vx,max_vx,min_vx,mean_vy,sd_vy,max_vy,min_vy,mean_v,sd_v,max_v,min_v,mean_a,sd_a,max_a,min_a,mean_jerk,sd_jerk,max_jerk,min_jerk,a_beg_time,userid"+\
                    "\n"

ACTION_CSV_HEADER = [BALABIT_ACTION, CHAOSHEN_ACTION, TWOS_ACTION, DFL_ACTION]




ACTION_CSV_HEADER_LEGALITY = ["type_of_action,traveled_distance_pixel,elapsed_time,direction_of_movement,straightness,num_points,sum_of_angles,mean_curv,sd_curv,max_curv,min_curv," \
                    "mean_omega,sd_omega,max_omega,min_omega,largest_deviation,dist_end_to_end_line,num_critical_points,"+\
                    "mean_vx,sd_vx,max_vx,min_vx,mean_vy,sd_vy,max_vy,min_vy,mean_v,sd_v,max_v,min_v,mean_a,sd_a,max_a,min_a,mean_jerk,sd_jerk,max_jerk,min_jerk,a_beg_time,userid,legality"+\
                    "\n", "type_of_action,traveled_distance_pixel,elapsed_time,direction_of_movement,straightness,num_points,sum_of_angles,mean_curv,sd_curv,max_curv,min_curv," \
                    "mean_omega,sd_omega,max_omega,min_omega,largest_deviation,dist_end_to_end_line,num_critical_points,"+\
                    "mean_vx,sd_vx,max_vx,min_vx,mean_vy,sd_vy,max_vy,min_vy,mean_v,sd_v,max_v,min_v,mean_a,sd_a,max_a,min_a,mean_jerk,sd_jerk,max_jerk,min_jerk,a_beg_time,userid,legality"+\
                    "\n"]

OUTPUT_FILE = 'processed_files'

## SCREEN LIMIT ##

X_THRESHOLD = 4000
Y_THRESHOLD = 4000


TEST_FILES = "test_files\\Singapur"