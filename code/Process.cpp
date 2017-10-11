#include<iostream>
#include"Preprocess.h"

#include <math.h>
#include <stdlib.h>
#include <core.hpp>
#include <highgui.hpp>
#include <imgproc.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"


using namespace std;
using namespace cv;

void Image::readImg(char *img) {													
	Imgin = cvLoadImage(img);														//读入图像
	cols = Imgin->width;															//宽度（列数）
	rows = Imgin->height;															//高度（行数）
	Imgout = cvCreateImage(cvGetSize(Imgin), IPL_DEPTH_8U, 3);						//同样3通道，大小一样，深度一样

	for (int i = 0; i < rows; ++ i) {                                               //赋值，使输出图像复制原图像
		uchar *ptrin = (uchar*)(Imgin->imageData + i * Imgin->widthStep);
		uchar *ptrout = (uchar*)(Imgout->imageData + i * Imgout->widthStep);
		for (int j = 0; j < Imgout->width; j ++) {
			ptrout[3 * j] = ptrin[3 * j];
			ptrout[3 * j + 1] = ptrin[3 * j + 1];
			ptrout[3 * j + 2] = ptrin[3 * j + 2];
		}
	}																				//把输出图像先赋值成输入图像
}

bool Image::ifreadFail() {                                                          //图像没load进来
	return Imgin == NULL;
}

void Image::showInput() {
	cvNamedWindow("input", CV_WINDOW_AUTOSIZE);										//创建窗口
	cvShowImage("input", Imgin);
	cout << "Close the image window, continue:" << endl;
	waitKey(0);																		//防止窗口一闪而过
}

void Image::selectMethod() {
	cout << endl << "ChooseMethod:'1' for nearest, '2' for bilinear, '3' for bicubic:" << endl;
	method = '1';																	//默认最近邻
	method = _getch();
	while (method != '1' && method != '2' && method != '3') {
		cout << "Inappropriate number,try again:" << endl;							//输入条件约束
		method = _getch();
	}
}

void Image::afterProc() {
	cvNamedWindow("", CV_WINDOW_AUTOSIZE);									//创建窗口
	cvShowImage("", Imgout);												//显示输出的图像
	waitKey(0);
	cout << endl << "Do you want to save the image？ y/n" << endl;
	char c ;
	c = _getch();
	while (c != 'n' && c != 'y') {
		cout << "Input error! Try again!";
		c = _getch();
	}
	if (c == 'y') {
		cout << endl << "please input your image's name of saving:";
		string imgname = "";
		cin >> imgname;
		imgname += ".jpg";
		const char *n = imgname.data();
		cvSaveImage(n, Imgout);	
		return;											//保存图像
	}
	cout << endl << "Close the image window, continue: " << endl << endl;
	cvWaitKey(0);																	//等待按键
	cvDestroyWindow("Result");														//销毁窗口
}

void Image::nearest(double x0, double y0, int u, int v) {                           //最近邻插值
																					//把原图离(x0,y0)最近的点的值赋给新图的(u,v)点
	int x = (int)(x0 + 0.5);
	int y = (int)(y0 + 0.5);														//+0.5之后强制类型转换就得到了最接近的坐标
	uchar *ptrout = (uchar*)(Imgout->imageData + (v)*Imgout->widthStep);
	uchar *ptrin = (uchar*)(Imgin->imageData + (y)*Imgin->widthStep);
	ptrout[3 * u] = ptrin[3 * x];													//把最近邻点的像素值赋给该点
	ptrout[3 * u + 1] = ptrin[3 * x + 1];
	ptrout[3 * u + 2] = ptrin[3 * x + 2];
}

void Image::bilinear(double x0, double y0, int u, int v) {                          //双线性插值 把img1(x0,y0)四周四个点经双线性运算赋值给img2(u,v)
	int x = (int)x0;
	int y = (int)y0;

	double k1 = (double)(x0 - x);
	double k2 = (double)(y0 - y);
	double tmp1, tmp2;

	uchar *ptrout = (uchar*)(Imgout->imageData + (v)*Imgout->widthStep);
	uchar *ptrin = (uchar*)(Imgin->imageData + (y)*Imgin->widthStep);				//ptrin是y这一列，ptrin[3 * x]是左上，ptrin[3 * (x + 1)]左下
	uchar *ptrin2 = (uchar*)(Imgin->imageData + (y + 1) * Imgin->widthStep);		//ptrin2是y的下一列，右上和右下

	for (int k = 0; k < 3; ++ k) {
		tmp1 = (1 - k1) * ptrin[3 * x + k] + k1 * ptrin[3 * (x + 1) + k];			//对应(i，j)和(i+1,j)的加权中值
		tmp2 = (1 - k1) * ptrin2[3 * x + k] + k1 * ptrin2[3 * (x + 1) + k];			//对应(i,j+1)和(i+1,j+1)的加权中值
		ptrout[3 * u + k] = (uchar)((1 - k2) * tmp1 + k2 * tmp2);					//对应上2个中值的插值
	}
}

double S(double x) {																//双三次插值核函数
	x = abs(x);
	if (x <= 1)  return 1 - 2 * x * x + x * x * x;
	else if (x <= 2) return 4 - 8 * x + 5 * x * x - x * x * x;
	else return 0;
}

int pos(int x) {
	if (x < 0) return 0;
	else return x;
}

void Image::bicubic(double x0, double y0, int u, int v) {                            
	int x = (int)x0;
	int y = (int)y0;																	//整数部分
	double dx = (double)(x0 - x);
	double dy = (double)(y0 - y);														//小数部分
	double A[4] = { 0.0 }, C = 0.0;
	uchar *ptry[4] = { NULL };															//像素指针

	uchar *ptrout = (uchar*)(Imgout->imageData + v * Imgout->widthStep);
	ptry[0] = (uchar*)(Imgin->imageData + (y - 1) * Imgin->widthStep);					//上一列
	ptry[1] = (uchar*)(Imgin->imageData + y * Imgin->widthStep);						//这一列
	ptry[2] = (uchar*)(Imgin->imageData + (y + 1) * Imgin->widthStep);					//下一列
	ptry[3] = (uchar*)(Imgin->imageData + (y + 2) * Imgin->widthStep);					//下下列

	for (int k = 0; k < 3; ++ k) {
		C = 0;
		for (int col = 0; col < 4; ++ col){ 
			A[col] = S(dx + 1) * (double)ptry[col][3 * (x - 1) + k] + S(dx) * (double)ptry[col][3 * x + k] + S(dx - 1) * (double)ptry[col][3 * (x + 1) + k]  + S(dx - 2) * (double)ptry[col][3 * (x + 2) + k];
			C += S(dy + 1 - col) * A[col];
		}
		if (C > 255)ptrout[3 * u + k] = 255;											//确保像素值在1~255范围内
		else if (C < 1) ptrout[3 * u + k] = 1;
		else ptrout[3 * u + k] = (uchar)C;	
	}
}

void Image::preRotate() {
	cout << endl << "Your image size:" << rows << " rows * " << cols << " cols " << endl;
	do {
		cout << endl << "Enter the process center:" << endl << "vertCenter =";
		cin >> vertCenter;
		cout << endl << "horiCenter =";
		cin >> horiCenter;
		cout << endl << "Input the radius of rotate:" ;
		cin >> rotateR;
		if ((horiCenter + rotateR) > cols-1 || (horiCenter - rotateR) < 0
			|| (vertCenter + rotateR) > rows-1 || (vertCenter - rotateR) < 0) {
			cout << "Inappropriate number,try again:" << endl;
		}
	} while ((horiCenter + rotateR) > cols-1 || (horiCenter - rotateR) < 0 || (vertCenter + rotateR) > rows-1 || (vertCenter - rotateR) < 0);

	selectMethod();

	cout << endl << "Input the rotate angle(degree:°)(negative allowed): ";
	cin >> rotateAngle;
}

void Image::rotate() {
	preRotate();
	double row = rotateR;																//最大旋转半径
	double radius;																		//极坐标中离原点的距离r
	int i, j;
	for (i = -row; i < row; ++ i) {														//i,j 分别是X,Y方向上离中心点的距离
		for (j = -row; j < row; ++ j) {
			radius = sqrt(double(j * j + i * i));
			if (radius <= row) {
				beta = PI * rotateAngle / 180 * (1 - radius / row);
				int u = j + horiCenter;													//待处理点纵向坐标
				int v = i + vertCenter;													//待处理点横向坐标
				double x0 = (double)((j)*cos(beta) - (i)*sin(beta) + horiCenter);		//映射后纵向坐标
				double y0 = (double)((j)*sin(beta) + (i)*cos(beta) + vertCenter);		//映射后横向坐标
				if (method == '1') nearest(x0, y0, u, v);
				else if (method == '2') bilinear(x0, y0, u, v);
				else if (method == '3') bicubic(x0, y0, u, v);
			}
		}
	}
}

void Image::preDistorsion() {
	cout << endl << "Your image size：" << rows << " rows * " << cols << " cols " << endl;
	horiCenter = cols / 2;
	vertCenter = rows / 2;

	selectMethod();

	cout << "Bigger or smaller, b/s?" << endl;
	biggerOrSmaller = _getch();
	while (biggerOrSmaller != 'b' && biggerOrSmaller != 's') {
		cout << "Input error! Please input again:"<<endl;
		biggerOrSmaller = _getch();
	}
}

double solveFunction(double x, double ru, double eps = 1e-6) {                      // solve "kx^3 + x = c", k = 0.000005
	double y = 0.000005 * x * x * x + x - ru;
	if (abs(y) < eps) return x;
	return solveFunction(x - y / (3 * 0.000005 * x * x + 1), ru, eps);
}

void Image::distorsion() {                                     //畸变函数
	preDistorsion();
	if (biggerOrSmaller == 'b') {                              //桶形畸变，看上去有放大的效果
		for (int i = 0; i < cols; ++ i) {
			for (int j = 0; j < rows; ++ j) {
				ru = sqrt((i - horiCenter) * (i - horiCenter) + (j - vertCenter) * (j - vertCenter));
				double rd = solveFunction(0, ru);
				double alpha = 0;

				if (i < horiCenter)  alpha= atan((double)(-j + vertCenter) / (i - horiCenter)) + PI;
				else alpha = atan((double)(-j + vertCenter) / (i - horiCenter));
				
				double x0 = rd * cos(alpha) + horiCenter;
				double y0 = rd * sin(alpha) + vertCenter;

				if (x0 >= 0 && x0 < cols && y0 >= 0 && y0 < rows) {    
					if (method == '1') nearest(x0, rows-y0, i, j);
					else if (method == '2') bilinear(x0, rows - y0, i, j);
					else if (method == '3') bicubic(x0, rows - y0, i, j);
				}else {																	//超范围的设置为0
					(Imgout->imageData + (j)*Imgout->widthStep)[3 * i + 0] = 0;
					(Imgout->imageData + (j)*Imgout->widthStep)[3 * i + 1] = 0;
					(Imgout->imageData + (j)*Imgout->widthStep)[3 * i + 2] = 0;
				}	
			}
		}
	}else {                                                    //枕形畸变，看上去有缩小的效果
		for (int i = 0; i < rows; ++ i) {
			for (int j = 0; j < cols; ++ j) {
				int dy = (i - vertCenter);
				int dx = (j - horiCenter);
				double r = (double)(dx * dx + dy * dy);
				int u = j;
				int v = i;
				double x0 = (double)(j + dx * (0.000005 * r));												//横坐标变换回原图像像素点坐标 
				double y0 = (double)(i + dy * (0.000005 * r));												//纵坐标变换回原图像像素点坐标

				if (x0 < 0 || x0 > cols - 1 || y0<0 || y0 > rows - 1) 
					for (int k = 0; k < 3; ++ k) (Imgout->imageData + v * Imgout->widthStep)[3 * u + k] = 0;
				else {
					if (method == '1') nearest(x0, y0, u, v);
					else if (method == '2') bilinear(x0, y0, u, v);
					else if (method == '3') {
						if (x0<1 || x0>cols - 3 || y0 < 1 || y0 > rows - 3) 
							for (int k = 0; k < 3; ++ k) (Imgout->imageData + v * Imgout->widthStep)[3 * u + k] = 0;
						else bicubic(x0, y0, u, v);
					}
				}
			}
		}
	}
}

double Ur(double r) {                               //径向基函数
	if (r != 0) return r * r * log( r * r);
	else return 0;
}

void Image::TPS() {	
	cout<<"processing...... Please wait with patience."<<endl;
	Mat img = cvarrToMat(Imgin);
	Mat imgout = cvarrToMat(Imgout);				//采用opencv中的Mat类来进行矩阵操作
	
	ctrlPoint controlPoint[5];									//5个控制点

	controlPoint[0].row = rows / 5;	
	controlPoint[0].col = cols / 5;							//0在左上

	controlPoint[1].row = rows / 5;	
	controlPoint[1].col = cols * 4 / 5;						//1在右上

	controlPoint[2].row = rows / 2;		
	controlPoint[2].col = cols / 2;							//2在中央

	controlPoint[3].row = 4 * rows / 5;		
	controlPoint[3].col = cols / 5;							//3在左下

	controlPoint[4].row = 4 * rows / 5;
	controlPoint[4].col = 4 * cols / 5;						//4在右下

	ctrlPoint dest[5];									//5个目标点

	dest[0].row = controlPoint[0].row - 50;
	dest[0].col = controlPoint[0].col - 35;
	dest[1].row = controlPoint[1].row - 61;
	dest[1].col = controlPoint[1].col + 40;
	dest[2].row = controlPoint[2].row + 48;
	dest[2].col = controlPoint[2].col - 57;
	dest[3].row = controlPoint[3].row + 43;
	dest[3].col = controlPoint[3].col - 43;
	dest[4].row = controlPoint[4].row + 58;
	dest[4].col = controlPoint[4].col - 30;

	Mat L = Mat::zeros(8, 8, CV_64F);				//初始化矩阵L(double)

	for (int i = 0; i < 5; ++i) {
		double *p = L.ptr<double>(i);
		for (int j = 0; j < 5; ++ j) 				//矩阵L中的左上角矩阵K
			if (j != i)  p[j] = Ur(sqrt((controlPoint[i].col - controlPoint[j].col) * (controlPoint[i].col - controlPoint[j].col) + (controlPoint[i].row - controlPoint[j].row)*(controlPoint[i].row - controlPoint[j].row)));
			
		p[5] = 1;									//右上角矩阵P
		p[6] = controlPoint[i].row;
		p[7] = controlPoint[i].col;
	}

	for (int j = 0; j < 5; ++j) {					//左下角矩阵：P的转置
		L.ptr<double>(5)[j] = 1;
		L.ptr<double>(6)[j] = controlPoint[j].row;
		L.ptr<double>(7)[j] = controlPoint[j].col;
	}												


	Mat Y = Mat::zeros(8, 2, CV_64F);				//矩阵Y(double)
	for (int i = 0; i < 5; i++) {
		double *p = Y.ptr<double>(i);
		p[0] = dest[i].row;
		p[1] = dest[i].col;
	}

	Mat ratio = (L.inv()) * Y;						//解线性方程组，直接用L的逆左乘Y

	Mat w[5], a1, ax, ay;					
	for (int i = 0; i < 5; ++i) w[i] = ratio.row(i);	//得到5个w向量
	
	a1 = ratio.row(5);
	ax = ratio.row(6);
	ay = ratio.row(7);

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			// 对每一个像素点用课件中f(x,y)的函数进行映射

			Mat temp = Mat::zeros(1, 2, CV_64F);

			for (int k = 0; k < 5; ++k) temp += w[k] * Ur(sqrt((i - controlPoint[k].row) * (i - controlPoint[k].row) + (j - controlPoint[k].col) * (j - controlPoint[k].col)));

			Mat xy = Mat::zeros(1, 2, CV_64F);
			xy = a1 + ax * i + ay * j + temp;

			double x0 = xy.ptr<double>(0)[0];
			double y0 = xy.ptr<double>(0)[1];

			if (x0 < 0 || x0 > rows - 1 || y0 < 0 || y0 > cols - 1) {    //超范围的设置为0
				imgout.at<Vec3b>(i, j)[0] = 0;
				imgout.at<Vec3b>(i, j)[1] = 0;
				imgout.at<Vec3b>(i, j)[2] = 0;
			}else {
				int x = (int)x0;
				int y = (int)y0;
				double dx = x0 - x;
				double dy = y0 - y;
				Mat B[3], rgb[3];										//因为rgb图像3通道，所以B也有3个元素
				int RGB[3];												
				Mat A = (Mat_<double>(1, 2) << (1 - dx), dx);		
				Mat C = (Mat_<double>(2, 1) << (1 - dy), dy);
				for (int k = 0; k < 3; ++k) {
					B[k] = (Mat_<double>(2, 2) << double(img.at<Vec3b>(x, y)[k]), double(img.at<Vec3b>(x, y + 1)[k]), double(img.at<Vec3b>(x + 1, y)[k]), double(img.at<Vec3b>(x + 1, y + 1)[k]));
					rgb[k] = A * B[k] * C;
					RGB[k] = (int)determinant(rgb[k]);
					imgout.at<Vec3b>(i, j)[k] = RGB[k];					//最后这里是向同学学习了Mat类的使用方法，尝试了用Mat类写了双线性插值
				}
			}
		}
	}
}