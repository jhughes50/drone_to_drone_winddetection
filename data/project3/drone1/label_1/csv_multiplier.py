# Jason Hughes
# July 25, 2019
# Program to multiply the number of label 1 data sets

import pandas as pd
import random as rd

#tstart = []
#tfinish = []
accx = []
accy = []
accz = []
gyrox = []
gyroy = []
gyroz = []
statex = []
statey = []
statez = []
stabip = []
stabir = []
stabiy = []


for n in range(3):
   for i in range(24):

      j = i + (n+1)*24

      df = pd.read_csv('data_set_label_1_packet_%i.csv' %i)

      #time_stamp_start = df['timestamp_start']
      #time_stamp_end = df['timestamp_end']
      ax = df['acc.x']
      ay = df['acc.y']
      az = df['acc.z']
      gx = df['gyro.x']
      gy = df['gyro.y']
      gz = df['gyro.z']
      sx = df['stateEstimate.x']
      sy = df['stateEstimate.y']
      sz = df['stateEstimate.z']
      stabp = df['stabilizer.pitch']
      stabr = df['stabilizer.roll']
      staby = df['stabilizer.yaw']
   
      for m in range(len(ax)):   

         #tstart.append(time_stamp_start[m])
         #tfinish.append(time_stamp_end[m])
         accx.append(ax[m])
         accy.append(ay[m])
         accz.append(az[m])
         gyrox.append(gx[m])
         gyroy.append(gy[m])
         gyroz.append(gz[m])
         statex.append(sx[m])
         statey.append(sy[m])
         statez.append(sz[m])
         stabip.append(stabp[m])
         stabir.append(stabr[m])
         stabiy.append(staby[m])

      dfn = pd.DataFrame()
      
      #dfn['timestamp_start'] = tstart
      #dfn['timestamp_end'] = tfinish
      dfn['acc.x'] = accx
      dfn['acc.y'] = accy
      dfn['acc.z'] = accz
      dfn['gyro.x'] = gyrox
      dfn['gyro.y'] = gyroy
      dfn['gyro.z'] = gyroz
      dfn['stateEstimate.x'] = statex
      dfn['stateEstimate.y'] = statey
      dfn['stateEstimate.z'] = statez
      dfn['stabilizer.pitch'] = stabip
      dfn['stabilizer.roll'] = stabir
      dfn['stabilizer.yaw'] = stabiy

      dfn.to_csv('data_set_label_1_packet_%i.csv' %j)

      #tstart.clear()
      #tfinish.clear()
      accx.clear()
      accy.clear()
      accz.clear()
      gyrox.clear()
      gyroy.clear()
      gyroz.clear()
      statex.clear()
      statey.clear()
      statez.clear()
      stabip.clear()
      stabir.clear()
      stabiy.clear()

