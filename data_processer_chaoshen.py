from utils.consts import *
import csv
import os
from utils.helper_functions import *
from data_processer import *

class DataProcesserChaoshen(DataProcesser):
    def __init__(self, user, limit):
        super().__init__(user, limit)
    
    def __map_seconds_to_database_format(self,seconds, min_val, max_val):
        # Assume that you have some pre-existing data for time and corresponding database values
        # Replace the following lists with your actual data

        # Map the provided seconds to the corresponding time using linear interpolation
        mapped_time = int(np.interp(seconds,  [0, 30 * 60], [min_val, max_val]))

        # Use the mapped time to interpolate the corresponding database value
        # database_value = np.interp(mapped_time, time_data, database_values)
        min_action_time = mapped_time - min_val
        return min_action_time

    
    def __computeFeatures(self, x, y, t, action, file, start, stop, user):
        lenght = len(x)
        if lenght < GLOBAL_MIN_ACTION_LENGHT_CHAOSHEN:             ## SHOULDNT EVER HAPPEN !!
            return None
        
        x = [int(n) for n in x]
        y = [int(n) for n in y]
        t = [float(n) for n in t]

        if containsNull(x): ## SCROLL ACTIONS
            return None
        
        for i in range(1, lenght):
            if (x[i] > X_THRESHOLD or y[i] > Y_THRESHOLD): ## DELETE THE OUT OF SCREEN SAMPLES
                x[i] = x[i - 1]
                y[i] = y[i - 1]
        # trajectory from the beginning point of the action
        # angles
        trajectory = 0
        sumOfAngles = 0
        angles = [ 0 ]
        path = [ 0 ]
        # velocities
        vx = [ 0 ]
        vy = [ 0]
        v = [ 0 ]
        for i in range(1, lenght):
            dx = int(x[i]) - int(x[i - 1])
            dy = int(y[i]) - int(y[i - 1])
            dt = float(t[i]) - float(t[i-1])
            if dt == 0:
                dt = 0.01  ## 
            vx_val = dx/dt
            vy_val = dy/dt
            vx.append(vx_val)
            vy.append(vy_val)
            v.append(math.sqrt(vx_val * vx_val +vy_val* vy_val))
            angle = math.atan2(dy,dx)
            angles.append( angle )
            sumOfAngles += angle
            distance = math.sqrt( dx * dx + dy *dy)
            trajectory = trajectory + distance
            path.append(trajectory)

        mean_vx = mean(vx, 0, len(vx))
        sd_vx = stdev(vx, 0, len(vx))
        max_vx = maximum(vx, 0, len(vx))
        min_vx = min_not_null(vx, 0, len(vx))

        mean_vy = mean(vy, 0, len(vy))
        sd_vy = stdev(vy, 0, len(vy))
        max_vy = maximum(vy, 0, len(vy))
        min_vy = min_not_null(vy, 0, len(vy))

        mean_v = mean(v, 0, len(v))
        sd_v = stdev(v, 0, len(v))
        max_v = maximum(v, 0, len(v))
        min_v = min_not_null(v, 0, len(v))

        # angular velocity
        omega = [0]
        no = len(angles)
        for i in range(1, no):
            dtheta = angles[ i ]-angles[ i-1 ]
            dt = float(t[i]) - float(t[i - 1])
            if dt == 0:
                dt = 0.01
            omega.append( dtheta/dt)

        mean_omega = mean(omega, 0, len(omega))
        sd_omega = stdev(omega, 0, len(omega))
        max_omega = maximum(omega, 0, len(omega))
        min_omega = min_not_null(omega, 0, len(omega))

        # acceleration
        a = [ 0 ]
        accTimeAtBeginning = 0
        cont = True
        for i in range(1, lenght - 1):
            dv = v[ i ] - v[ i-1 ]
            dt = float(t[i]) - float(t[i - 1])
            if dt == 0:
                dt = 0.01
            if cont  and dv > 0 :
                accTimeAtBeginning += dt
            else:
                cont = False
            a.append( dv/dt)

        mean_a = mean(a, 0, len(a) )
        sd_a = stdev(a, 0, len(a) )
        max_a = maximum(a, 0, len(a) )
        min_a = min_not_null(a, 0, len(a))

        # jerk
        j = [0]
        na = len(a)
        for i in range(1, na):
            da = a[i] - a[i - 1]
            dt = float(t[i]) - float(t[i - 1])
            if dt == 0:
                dt = 0.01
            j.append(da / dt)

        mean_jerk = mean(j, 0, len(j))
        sd_jerk = stdev(j, 0, len(j))
        max_jerk = maximum(j, 0, len(j))
        min_jerk = min_not_null(j, 0, len(j))

        # curvature: defined by Gamboa&Fred, 2004
        ## rzecz wzięta bezpośrednio z kodu autorów 
        # numCriticalPoints
        c = []
        numCriticalPoints = 0
        nn = len(path)
        for i in range(1, nn):
            dp = path[i]-path[i-1]
            if dp == 0:
                continue
            dangle = angles[i] - angles[i - 1]
            curv = dangle/dp
            c.append( curv )
            if abs(curv) < CURV_THRESHOLD:
                numCriticalPoints = numCriticalPoints + 1
        mean_curv = mean( c, 0, len(c))
        sd_curv = stdev(c, 0, len(c))
        max_curv = maximum(c, 0, len(c))
        min_curv = min_not_null(c, 0, len(c))

        # time
        time = float(t[lenght - 1]) - float(t[0])

        # direction: -pi..pi
        theta = math.atan2(int(y[lenght - 1]) - int(y[0]), int(x[lenght - 1]) - int(x[0]))
        direction = computeDirection(theta)

        # distEndtToEndLine
        distEndToEndLine = math.sqrt((int(x[0]) -int(x[lenght - 1])) * (int(x[0]) -int(x[lenght - 1])) +(int(y[0]) -int(y[lenght - 1]))*(int(y[0]) -int(y[lenght - 1])))

        # straightness
        if trajectory == 0:
            straightness = 0
        else:
            straightness =  distEndToEndLine / trajectory
        if straightness > 1:
            straightness = 1

        # largest deviation
        largest_deviation = largestDeviation(x,y)

        result = str(action) + ',' + str(trajectory) + ',' + str(time) + ',' + str(direction) + ','+\
                    str(straightness)+ ','+ str(lenght)+','+str(sumOfAngles)+','+\
                    str(mean_curv) + "," + str(sd_curv) + "," + str(max_curv) + "," +str(min_curv)+","+\
                    str(mean_omega)+","+str(sd_omega)+","+str(max_omega)+","+str(min_omega)+","+\
                    str(largest_deviation)+","+\
                    str(distEndToEndLine)+","+str(numCriticalPoints)+","+\
                    str(mean_vx)   + "," + str(sd_vx)   + "," +str(max_vx) + "," +str(min_vx)+","+\
                    str(mean_vy)   + "," + str(sd_vy)   + "," +str(max_vy) + "," +str(min_vy)+","+\
                    str(mean_v)    + "," + str(sd_v)    + "," +str(max_v)  + "," +str(min_v)+","+\
                    str(mean_a)    + "," + str(sd_a)    + "," + str(max_a) + "," +str(min_a)+","+\
                    str(mean_jerk) + "," + str(sd_jerk) + "," + str(max_jerk) + "," +str(min_jerk)+","+\
                    str(accTimeAtBeginning)+","+\
                    str(user)+\
                    "\n"
        return result


    def __queueAction(self, x, y, t, actionCode, action_file, n_from, n_to, user):
        result = self.__computeFeatures(x, y, t, actionCode, action_file, n_from, n_to, user) ## wykonaj liczenie feateruw
        if result != None:
            action_file.write(result) ## dopisz do pliku
        return
    #one MM action
    def __processMM(self, x, y, t, action_file, start, stop, user):
        # print("MM")
        self.__queueAction(x, y, t, MM, action_file, start, stop, user)
        return

    # one DD action
    def __processDD(self, x, y, t,action_file, start, stop, user):
        # print("DD")
        self.__queueAction(x, y, t, DD, action_file, start, stop, user)
        return

    # one SS action
    def __processSS(self, x, y, t, action_file, start, stop, use): # to jest do dodania
       # print("SS")
        # queueAction(x, y, t, DD, action_file, start, stop, user, is_legal)
        return
   
    def __processPC(self, x, y, t, action_file, start, stop, user): # to jest do dodania
       # print("SS")
        self.__queueAction(x, y, t, PC, action_file, start, stop, user)
        return
   
    def __processCombinedPC(self, actions, action_file, start, stop, user, time):
        x = []
        y = []
        t = []
        counter = 0
        lastTimestamp = 0
        for action in actions:
            event = action['event']
            currentTimestamp = float(action['t'])
            counter += 1
            if event == "Click()" or event == "RightClick()" or event == "DblClick()" or event == "RightClick()" or event == "RightDblClick()":
                if len(t) > GLOBAL_MIN_ACTION_LENGHT_CHAOSHEN: ##  if len is not sufficient then change
                    x.append(action['x'])
                    y.append(action['y'])
                    t.append(currentTimestamp)
                    self.__processPC(x, y, t, action_file, start, stop, user) ## save PC action
                return
            else:
                if currentTimestamp - lastTimestamp > time: ## TODO THIS REQUIRMENT WORKS ONLY FOR BALABIT, IT HAS TO BE CHANGED

                    stop = start + counter - 2 ## - 2 because the last 2 are release press
                    if len(t) > GLOBAL_MIN_ACTION_LENGHT_CHAOSHEN:
                        self.__processMM(x, y, t, action_file, start, stop, user) ## save PC action
                    x = []
                    y = []
                    t = []
                    start = stop + 1
                    lastTimestamp = currentTimestamp
                else:
                    x.append(action['x'])
                    y.append(action['y'])
                    t.append(currentTimestamp)
        return


    def __processCombinedDD(self, actions, action_file, start, stop, user, time):
        x = []
        y = []
        t = []
        counter = 0
        lastTimestamp = 0
        drag = False

        for action in actions:
            # state = action['state']
            # button = action['button']
            event = action['event']
            currentTimestamp = float(action['t'])
            counter += 1
            
          
            if event == "Click()" or event == "RightClick()" or event == "DblClick()" or event == "RightClick()" or event == "RightDblClick()" : ## END MM ACTION START DD
                stop = start + counter - 2
                if len(t) > GLOBAL_MIN_ACTION_LENGHT_CHAOSHEN:
                    self.__processMM(x, y, t, action_file, start, stop, user)
                ## STARTS DD 
                lastTimestamp = currentTimestamp
                drag = True
                x = []
                y = []
                t = []
                start = stop + 1
                x.append(action['x'])
                y.append(action['y'])
                t.append(currentTimestamp)

            if event == 'MouseEvent(WM_LBUTTONUP)' or event == 'MouseEvent(WM_RBUTTONUP)':
                # ends the DD action
                x.append(action['x'])
                y.append(action['y'])
                t.append(currentTimestamp)
                self.__processDD(x, y, t, action_file, start, stop, user)
                drag = False


            if event == "MouseEvent(WM_MOUSEMOVE)": ## SCAN MM ACTIONS  IF THEY RE LONG ENOUGH START ANOTHER ONE
                if drag:
                    x.append(action['x'])
                    y.append(action['y'])
                    t.append(currentTimestamp)
                else:
                    if currentTimestamp - lastTimestamp > time: ## TODO
                        stop = start + counter - 2
                        if len(t) > GLOBAL_MIN_ACTION_LENGHT_CHAOSHEN:
                            self.__processMM(x, y, t, action_file, start, stop, user) ## TODO THINK ABOUT ADDING MAX LENGHt
                            x = []
                            y = []
                            t = []
                            start = stop +1
                        lastTimestamp = currentTimestamp

                    x.append(action['x'])
                    y.append(action['y'])
                    t.append(currentTimestamp)

        return

            
                
    def createProcessedCSV(self, path , user, fileName, limit): ## check limit 
        start = 2
        end = 2
        counter = 1
        lastRow = None

        with open(path) as csvFile:
            amount = 0
            data = csv.DictReader(csvFile)
            # Get the last value
          # Get the first row
            first_row = next(data)
            min_time = int(first_row['TimeStamp'])
            # Get the last row
            for last_row in data:
                pass  # Let the loop iterate to the end
            max_time = int(last_row['TimeStamp'])

            # MAPPING THE TIME ## 
            time = self.__map_seconds_to_database_format(0.5, min_time, max_time)
            if time < GLOBAL_MIN_TIME_CHAOSHEN:
                time = 5000

            actions = []
            csvFile.seek(0)
            next(data)
            for row in data:
                if amount > limit:
                    break
                counter = counter + 1 # the counter where action starts and where it ends
                if lastRow != None and lastRow == row:
                    continue # skipping the duplicates (there is some of it in the data)
                
                ## TAKING CARE OF DUPLICATES  in dataset ##
                if (row['event'] == 'Click()' or row['event'] == 'DblClick()' or row['event'] == 'RightClick()') and lastRow != None:
                        if lastRow['event'] == row['event']:
                            continue

                ## UNKNOWN SHOULDBT BE PROCESSED  ALSO SCROLL EVENTS SHOULD BE IGNORED##
                if (row['event'] == 'Unknown' or row['event'] == 'MouseEvent(WM_MBUTTONUP)' or row['event'] == 'MiddleDlbClick()' or row['event'] == 'MiddleClick()'):
                    continue ## TO ADD LATER 
                
                ## PROCESSING actions ## 
                record = {
                    "x": row['x'],
                    "y": row['y'],
                    "t": row['TimeStamp'],
                    "event": row['event'],
                }                 
               
                ## actions ##
               
                if row['event'] == 'MouseEvent(WM_LBUTTONUP)' or row['event'] == 'MouseEvent(WM_RBUTTONUP)': ## create EVENT ## CLICK DOUBlE CLICK
                    actions.append(record)
                    if len(actions) <= GLOBAL_MIN_ACTION_LENGHT_CHAOSHEN: ## Restart the data structures, because the action is too short (maybe random)
                        actions = []
                        start = counter
                        continue

                    if lastRow!= None and lastRow['event'] == 'MouseEvent(WM_MOUSEMOVE)':
                        end = counter
                        self.__processCombinedDD(actions, fileName, start, end, user, time)
                        amount += 1

                    if (lastRow != None and (lastRow['event'] == "Click()" or (lastRow['event'] == "DblClick()"))) : ## PC or MM Action
                        end = counter
                        self.__processCombinedPC(actions, fileName, start, end, user, time)
                        amount += 1

                ## Processed --> start new action
                    actions = []
                    start = end + 1
                else:
                    if int(record['x'])< X_THRESHOLD or int(record['y']) < Y_THRESHOLD: ## TODO shold be "AND"
                        actions.append(record)
                    lastRow = row
            end = counter # THE LAST ITERATION IF THERE IS NO RELEASE 
            if amount > limit:
                return ## TO DO

            self.__processCombinedPC(actions, fileName, start, end, user, time)
            # actions, action_file, start, stop, user)
            amount +=1
            return


