# -*- coding: UTF-8 -*-
#各种模块导入
import cv2  
import numpy as np
import cv2.cv as cv
import math
import time



#读取图片
def readp():
 img1 = cv2.imread('image1.jpg')  
 return img1

#roi函数
def roi_mask(img, vertices):#读入canny轮廓图
  mask = np.zeros_like(img)
  mask_color = 255 #白色线
  cv2.fillPoly(mask, vertices, mask_color)#白色填充
  masked_img = cv2.bitwise_and(img, mask)#加和
  return masked_img



def tc(img,lines):
  k=[]
  i=0
  for line in lines:
    q=0
    for x1, y1, x2, y2 in line:
     x1=float  (x1)
     x2=float  (x2)
     y1=float  (y1)
     y2=float  (y2)
     i=i+1
     if (x1==0 and x2==0) or x1==x2 :
      picture()
     p=(y2 - y1) / (x2 - x1)
     q=q+p
     k.append(p) 
  k1=q/i
  print '斜率总和',q,'平均斜率',k1
  
 
 
  

#标准霍夫变换参数全局变量
rho = 1
theta = np.pi / 180  
threshold = 15

min_line_length = 40
max_line_gap = 20

#霍夫直线函数
def draw_lines(img, lines, color=[0,0,255], thickness=3):#画红线
  for line in lines:
    for x1, y1, x2, y2 in line:
      cv2.line(img, (x1, y1), (x2, y2), color, thickness)
  #print x1,x2,y1,y2
def hough_lines(img, rho, theta, threshold, #霍夫变换
                min_line_len, max_line_gap):
  lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                          minLineLength=min_line_len,  
                          maxLineGap=max_line_gap)
  line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
  draw_lines(line_img, lines)
  
  cv2.imwrite("line_img.jpg",line_img)
  #tc(line_img, lines)
 

  return line_img
   
#画拟合直线函数
def lines2():
 im=cv.LoadImage('roi_edges.jpg', cv.CV_LOAD_IMAGE_GRAYSCALE)
 pi = math.pi
 x = 0
 dst = cv.CreateImage(cv.GetSize(im), 8, 1)
 cv.Canny(im, dst, 200, 200)
 cv.Threshold(dst, dst, 100, 255, cv.CV_THRESH_BINARY)
 color_dst_standard = cv.CreateImage(cv.GetSize(im), 8, 3)
 cv.CvtColor(im, color_dst_standard, cv.CV_GRAY2BGR)#Create output image in RGB to put red lines
 lines = cv.HoughLines2(dst, cv.CreateMemStorage(0),    cv.CV_HOUGH_STANDARD, 1, pi/100, 71, 0, 0)
 klsum=0
 klaver=0
 krsum=0
 kraver=0

 #global k
 #k=0
 for (rho, theta) in lines[:100]:
    kl=[]
    kr=[]
    a = math.cos(theta) 
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (cv.Round(x0 + 1000*(-b)), cv.Round(y0 + 1000*(a)))
    pt2 = (cv.Round(x0 - 1000*(-b)), cv.Round(y0 - 1000*(a)))
    k=((y0 - 1000*(a))- (y0 + 1000*(a)))/((x0 - 1000*(-b))-(x0 +1000*(-b)))
 
    if  abs(k)<0.4:
      pass
    elif k>0:
       kr.append(k)
       len_kr=len(kr)
       for i in kr:
         krsum=krsum+i
         kraver=krsum/len_kr

         cv.Line(color_dst_standard, pt1, pt2, cv.CV_RGB(255, 0, 0), 2, 4)
    elif k<0:
       kr.append(k)
       kl.append(k)
       len_kl=len(kl)
       for i in kl:
         klsum=klsum+i
         klaver=klsum/len_kl
         cv.Line(color_dst_standard, pt1, pt2, cv.CV_RGB(255, 0, 0), 2, 4)
     #print k
  #  cv.Line(color_dst_standard, pt1, pt2, cv.CV_RGB(255, 0, 0), 2, 4)
 cv.SaveImage('lane.jpg',color_dst_standard)
 print '左车道平均斜率:',klaver  ,'  右车道平均斜率:',kraver
 cv.ShowImage("Hough Standard", color_dst_standard)
 cv.WaitKey(0)



#图片处理函数
def picture():
 
 img2=readp()
 imgGray1 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
 cv2.imwrite("gray.jpg", imgGray1)
 imgGray =cv2.equalizeHist( imgGray1 )
 cv2.imwrite("dst.jpg", imgGray)
 blur_ksize = 5  # Gaussian blur kernel size
 blur_gray = cv2.GaussianBlur(imgGray1, (blur_ksize, blur_ksize), 0, 0)
 canny_lthreshold = 0  # Canny edge detection low threshold
 canny_hthreshold = 70 # Canny edge detection high threshold
 edges = cv2.Canny(blur_gray,canny_lthreshold,canny_hthreshold)
 cv2.imwrite("edges.jpg", edges)
 roi_vtx = np.array([[(0, 300), (140,240), #四点画区域shape（0）为高
                     (600, 290), (img2.shape[1], img2.shape[0])]])
 roi_edges = roi_mask(edges, roi_vtx)#切割图像
 cv2.imwrite("roi_edges.jpg", roi_edges)
 lines2()
 line1_img = hough_lines(roi_edges, rho, theta, threshold, 
                       min_line_length, max_line_gap)
 
 cv2.imwrite("line_img.jpg",line1_img)

picture()
