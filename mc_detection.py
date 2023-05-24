import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def imshow(img):
    plt.figure(dpi=200)
    plt.imshow(img)
    plt.show

def cal_dis_p(p1, p2):
    return pow((p1[0]-p2[0]) ** 2 + (p1[1]-p2[1]) ** 2, 0.5)
  
  
# use a 45 degree straight line below the R angel to model distance
def mc_det(img, mc_thresh, mc_thresh_last, num_pts=7):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = gray.shape[0]
    w = gray.shape[1]
#     print(gray.shape)
    res, thresh = cv2.threshold(gray, 44, 255, cv2.THRESH_BINARY)
#     imshow(img)
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
#     imshow(edges)
    
    if (len(np.where(edges>250)[0]) < 100):
        return 0, 0

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength =20,  maxLineGap = 100)
    lmaxx, maxy = 0, img.shape[0]
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x1 - x2) < 20 and y1>300 and y2>300:
            maxy = min(maxy, min(y1,y2))
        if abs(y1 - y2) < 20:
            lmaxx = max(lmaxx, x2)
    print('before maxy', maxy)
    print('before lmaxx', lmaxx)
    
    if abs(lmaxx-maxy)<130:    
#         maxy = min(1100, maxy)
        lmaxx = min(800,lmaxx)
    edges[maxy:,:] = 0
    edges[:,:lmaxx] = 0
#     imshow(edges)
#     print('after maxy', maxy)
#     print('after lmaxx', lmaxx)    

    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges)
    _, times= np.unique(labels, return_counts = True)
    tt = times.argsort()[::-1]
    print(len(tt))
    mc = []
    previous_dist = []
    if len(tt)!=0:
        top_edge = np.array([np.where(labels == min(tt[1:3]))])
        bottom_edge = np.array([np.where(labels == max(tt[1:3]))])

        top_edge = top_edge.transpose(2,1,0).squeeze() # N,2
        bottom_edge = bottom_edge.transpose(2,1,0).squeeze()
        min_be, max_be = bottom_edge[0], bottom_edge[-1]
#         print(top_edge)
        
        
        
        
#         cnt = 0
        for i in range(top_edge.shape[0]):
            this_dist = 0
            point = top_edge[i]
            for dis in range(1, 100):
                move_point = [point[0] + dis, point[1] - dis]
                find_point = (bottom_edge[:,0] == move_point[0]) & (bottom_edge[:,1] == move_point[1])
                res = bottom_edge[find_point]
#                 cnt += 1
                if i >= num_pts:
                    last_dist = previous_dist[i-num_pts]
                else:
                    last_dist = 0
                if len(res) > 0 and dis > mc_thresh and last_dist!=0 and abs(dis-last_dist) > mc_thresh_last:
                    mc.append(point)
                    this_dist = dis
#                         cnt = 0
                    break
                if len(res) > 0:
                    this_dist = dis
                    break
            previous_dist.append(this_dist)
                
    #         if len(res) == 0:
    #             dis = min(caculate_dis(point - min_be), caculate_dis(point - max_be))
    #             if dis > mc_thresh:
    #                 mc.append(point)
        mc = np.array(mc)
        if mc.shape[0] < 1:
            print('No mc detected!')
#             imshow(edges)
            blank_img = np.zeros((h,w), dtype=np.float32)
            blank_img[top_edge[:,0],top_edge[:,1]] = 1
    #         imshow(blank_img)
            blank_img[bottom_edge[:,0],bottom_edge[:,1]] = 1
            imshow(blank_img)
            return 0, 0
        
        
        
        img[mc[:,0],mc[:,1]] = (255,0,0)
        print('Mc detected! show.')
        imshow(img)
    return mc, 1


# use a simple circle to model the R angel
def mc_det_w_translate(img, mc_thresh):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = gray.shape[0]
    w = gray.shape[1]
    res, thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)
#     imshow(img)
#     imshow(gray)
#     imshow(thresh)
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
#     imshow(edges)
    
    if (len(np.where(edges>250)[0]) < 100):
        return 0, 0

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength =20,  maxLineGap = 100)
    lmaxx, maxy = 0, img.shape[0]
    lmaxx_y = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x1 - x2) < 20 and y1>300 and y2>300:
            maxy = min(maxy, min(y1,y2))
        if abs(y1 - y2) < 20:
            if x2 > lmaxx:
                lmaxx_y = y2
            lmaxx = max(lmaxx, x2)
            
    print('before maxy', maxy)
    print('before lmaxx', lmaxx)
    
    if abs(lmaxx-maxy)<130:    
#         maxy = min(1100, maxy)
        lmaxx = min(800,lmaxx)
    edges[maxy:,:] = 0
    edges[:,:lmaxx] = 0
#     imshow(edges)
#     print('after maxy', maxy)
#     print('after lmaxx', lmaxx)    

    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges)
    _, times= np.unique(labels, return_counts = True)
    tt = times.argsort()[::-1]
    print(len(tt))
#     print(tt)
#     print(times)
#     print(labels)
    mc = []
    if len(tt)!=0:
        top_edge = np.array([np.where(labels == min(tt[1:3]))])
        top_edge = top_edge.transpose(2,1,0).squeeze() # N,2
        first_point = top_edge[0]
        
        translate_x = first_point[0] - 0
#         translate_x = lmaxx_y - 0
        translate_y = first_point[1] - 0
#         translate_y = lmaxx - 0
        print('translation xy: ',translate_x,', ', translate_y)
#         print(top_edge.shape)
        trans_top_edge = top_edge.copy()
        trans_top_edge[:,0] = top_edge[:,0] - translate_x
        trans_top_edge[:,1] = top_edge[:,1] - translate_y
        last_point = trans_top_edge[-1]
        r = (last_point[0]**2 + last_point[1]**2)/(2*last_point[0])
        print(r)
#         print(trans_top_edge.shape)
        for i in range(top_edge.shape[0]):
            point = trans_top_edge[i]
            dist = pow((point[0]-r) ** 2 + point[1] ** 2, 0.5)
#             dist = caculate_dis(point)
            if dist-r > mc_thresh:
                mc.append(top_edge[i])
#                 mc.append(point)

        mc = np.array(mc)
        if mc.shape[0] < 1:
            print('No mc detected!')
#             imshow(edges)
            blank_img = np.zeros((h,w), dtype=np.float32)
            blank_img[trans_top_edge[:,0],trans_top_edge[:,1]] = 1
            imshow(blank_img)
            return 0, 0
        
        
        
        img[mc[:,0],mc[:,1]] = (255,0,0)
        print('Mc detected! show.')
        imshow(img)
    return mc, 1


# use opencv to try to fit the R angle into a circle (not stable now)
def mc_det_circle_fit(img, mc_thresh):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = gray.shape[0]
    w = gray.shape[1]
    res, thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)
#     imshow(img)
#     imshow(gray)
#     imshow(thresh)
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
#     imshow(edges)
    
    if (len(np.where(edges>250)[0]) < 100):
        return 0, 0

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength =20,  maxLineGap = 100)
    lmaxx, maxy = 0, img.shape[0]
    lmaxx_y, maxy_x = 0, img.shape[1]
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x1 - x2) < 20 and y1>300 and y2>300:
            if min(y1, y2) < maxy:
                maxy = min(y1, y2)
                if y1 < y2:
                    maxy_x = x1
                else:
                    maxy_x = x2
#             maxy = min(maxy, min(y1,y2))
        if abs(y1 - y2) < 20:
            if x2 > lmaxx:
                lmaxx = x2
                lmaxx_y = y2
#     print('before maxy', maxy)
#     print('before lmaxx', lmaxx)
    
    if abs(lmaxx-maxy)<130:    
#         maxy = min(1100, maxy)
        lmaxx = min(800,lmaxx)
    edges[maxy:,:] = 0
    edges[:,:lmaxx] = 0
#     imshow(edges)
#     print('after maxy', maxy)
#     print('after lmaxx', lmaxx)
    print('maxy_x, maxy', maxy_x, maxy)
    print('lmaxx, lmaxx_y', lmaxx, lmaxx_y)

    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges)
    _, times= np.unique(labels, return_counts = True)
    tt = times.argsort()[::-1]
    print(len(tt))
#     print(tt)
#     print(times)
#     print(labels)
    mc = []
    if len(tt)!=0:
        top_edge = np.array([np.where(labels == min(tt[1:3]))])
        top_edge = top_edge.transpose(2,1,0).squeeze() # N,2
        first_point = top_edge[10]
        # fitting circle from three points on the arch
#         x1, y1 = maxy_x, maxy
#         x2, y2 = lmaxx, lmaxx_y
        x1, y1 = top_edge[0][1], top_edge[0][0]
        x2, y2 = top_edge[-1][1], top_edge[-1][0]
        x3, y3 = first_point[1], first_point[0]
#         print(f'{x1}, {y1}')
        m1 = (y2 - y1) / (x2 - x1)
        m2 = (y3 - y2) / (x3 - x2)
        print(m1, m2)
        x = (m1 * m2 * (y1 - y3) + m2 * (x1 + x2) - m1 * (x2 + x3)) / (2 * (m2 - m1))
        y = (-1 / m1) * (x - (x1 + x2) / 2) + (y1 + y2) / 2
        r = cal_dis_p([x1,y1], [x,y])
        print(f'The center of fitted circle is: {x}, {y}, radius is: {r}')
#         print(top_edge[45][1] , top_edge[45][0])
#         print(cal_dis_p([x2,y2], [x,y]))
#         print(cal_dis_p([x3,y3], [x,y]))
        for i in range(top_edge.shape[0]):
            point = top_edge[i]
            dist = cal_dis_p(point, [x,y])
#             print(dist-r)
            if dist-r > mc_thresh:
                mc.append(point)

        mc = np.array(mc)
        if mc.shape[0] < 1:
            print('No mc detected!')
#             imshow(edges)
            blank_img = np.zeros((h,w), dtype=np.float32)
            blank_img[top_edge[:,0],top_edge[:,1]] = 1
#             blank_img[y, x] = 1
            cv2.circle(blank_img, (int(x), int(y)), 10, 1, 3)
            imshow(blank_img)
            return 0, 0

        img[mc[:,0],mc[:,1]] = (255,0,0)
        print('Mc detected! show.')
        imshow(img)
    return mc, 1
  
if __name__ == '__main__':
  root = '/opt/ml/code/R_maoci/MC'
  root_path = os.listdir(root)
  total_cnt = 0
  true_cnt = 0
  for img_path in root_path:
      if os.path.splitext(img_path)[1] in ['.jpg', '.png']:
          total_cnt += 1
          print(os.path.join(root, img_path))
          img = cv2.imread(os.path.join(root, img_path))
  #         mc, stat = mc_det(img, mc_thresh=55, mc_thresh_last=0, num_pts=5)
          mc, stat = mc_det_w_translate(img, mc_thresh=6.5)
  #         mc, stat = mc_det_circle_fit(img, mc_thresh=7)
          if stat:
             true_cnt += 1
  #     break

  print('Total number of MC examples is: ', total_cnt)
  print('Detected MC examples is: ', true_cnt, ', detection rate: ',(true_cnt/total_cnt))
