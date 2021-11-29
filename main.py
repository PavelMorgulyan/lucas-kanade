import numpy as np
import cv2


# In[9]:


from scipy.spatial.distance import euclidean
from collections import defaultdict
import time
import sys
import gc

#включение эффекта
my_feature = True
try:
    if sys.argv[1] == "skeleton":
        cap = cv2.VideoCapture('Skeleton.mp4')
except:
    cap = cv2.VideoCapture('Woman.mp4')

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(1000,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

# Get new val
if my_feature:
    new_g_p = []
    for p in p0:
        x, y = p[0][0], p[0][1]
        new_g_p.extend([[[x+15, y]], [[x-15, y]], [[x, y-15]], [[x, y+15]]])
        break
    p0=np.float32(np.append(p0, np.array(new_g_p), axis=0))
    pnts = defaultdict(lambda: -1)
    pnts_g = []
    pnts_g_sc = []

    dist = defaultdict(lambda: {'x':0, 'y':0})

#j - кол кадров, в какой момент проверяем точки
j=0
while(1):
    j += 1
    if my_feature:
        if j == 40:
            print ("Phase moment")
            new_g_p = []
            max_x, max_y = 0, 0
            for pnt in dist:
                if dist[pnt]['x'] > max_x:
                    max_x = dist[pnt]['x']
                if dist[pnt]['y'] > max_y:
                    max_y = dist[pnt]['y']
            j = 0
            # 10 - минимальное кол "хороших" перемещений точек
            for pnt in pnts:
                if pnts[pnt] > 10 and pnts[pnt] > -1:
                     pnts_g.append(pnt)
                elif pnt in pnts_g_sc:
                    pnts_g.remove(pnt)
                    pnts_g_sc.remove(pnt)
                elif pnt in pnts_g:
                    pnts_g.remove(pnt)
                    # prediction: x_offset = max_x, y_offset = 0
                    new_point = np.add(pnt, (max_x, 0))
                    new_g_p.append(tuple(new_point))
                    pnts_g.append(tuple(new_point))
                    pnts_g_sc.append(tuple(new_point))
                pnts[pnt] = 0
            if new_g_p:
                new_g_p = np.array(new_g_p).reshape(-1,1,2)
                p0=np.float32(np.append(p0, new_g_p, axis=0))
            dist = defaultdict(lambda: {'x':0, 'y':0})
            pnts = defaultdict(lambda: -1)
            #for pnt in dist:
            #    dist[pnt]['x'], dist[pnt]['y'] = 0, 0
            # gc.collect()
    ret,frame = cap.read()
    if not ret:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        if my_feature:
            if (euclidean(new, old)) > .4 and a-c > 3: # thr > .3 / a-c = x_off on board:
                pnts[(a,b)] = pnts[(c,d)] + 1 if pnts[(c,d)] != -1 else 1
                dist[(a,b)]['x'] = max(dist[(c,d)]['x'], a-c, key=abs)
                dist[(a,b)]['y'] = max(dist[(c,d)]['y'], b-d, key=abs)
            else:
                pnts[(a,b)] = pnts[(c,d)]

            pnts[(c,d)] = -1
            dist[(c,d)]['x'], dist[(c,d)]['y'] = 0, 0

            if (c,d) in pnts_g_sc:
                pnts_g_sc.remove((c,d))
                pnts_g_sc.append((a,b))

            if (c,d) in pnts_g:
                pnts_g.remove((c,d))
                pnts_g.append((a,b))
                mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        else:
            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)

    cv2.imshow('frame', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()
