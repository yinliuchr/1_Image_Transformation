#pragma once

#include<iostream>
#include<string>
#include<opencv.hpp>
#include <cmath>
#include "cv.h"
#include "highgui.h"
#include<conio.h>

#define PI 3.1415926

using namespace std;
using namespace cv;

class Image {
	private:
		IplImage* Imgin;									//指向输入图像
		IplImage* Imgout;									//指向输出的图像
		int rows, cols;										//图像的宽和高
		int horiCenter, vertCenter;							//旋转中心点
		int rotateR;										//旋转半径                                
		char method ;										//插值方法,默认为最近邻
		double radius;										//处理点离处理中心的距离
		double rotateAngle ;								//中心点旋转角度
		double beta;										//转动角
		char biggerOrSmaller;
		double ru, eps;										//ru畸变的目标距离，eps很小的一个值
	public:
		Image() {											//构造函数
			Imgin = Imgout = NULL;
			rows = cols = 0;
			horiCenter = vertCenter = 0;
			rotateR = 0;
			method = '1';
			radius = rotateAngle = beta = 0;
		}
		~Image() {											//析构函数
			cvReleaseImage(&Imgin);                        
			cvReleaseImage(&Imgout);                      
		}
		bool ifreadFail();										//是否读入图像失败
		void showInput();										//显示输入图像
		void readImg(char *img);								//读入图像
		void selectMethod();									//选择插值方法
		void afterProc();										//是否保存处理后的
		void nearest(double x, double y, int u, int v);			//最近邻插值
		void bilinear(double x, double y, int u, int v);		//双线性插值
		void bicubic(double x, double y, int u, int v);			//双三次插值
		void preRotate();										//旋转预处理	
		void rotate();											//旋转
		void preDistorsion();									//畸变预处理
		void distorsion();										//畸变
		void TPS();												//TPS
		
};

struct ctrlPoint {										//像素点,TPS中控制点、目标点的结构
	int row;											//行数
	int col;											//列数
};