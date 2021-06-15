#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <random>
using namespace std;
using namespace cv;

#define NUMBER 1500

std::string imagesPath = "/home/demo/桌面/DL/Line-regression/dataset/test_images/";
std::string labelPath = "/home/demo/桌面/DL/Line-regression/dataset/label/"; 

int main(int argc, char const *argv[])
{
	ofstream outfile(labelPath+"test.txt", ios::out); 
	srand((int)time(0));
	for (size_t i=0; i<NUMBER; ++i){
		cv::Mat image = cv::Mat::zeros(cv::Size(224,224),CV_8UC1);
		// \rho = [-112,112]
		int rho_ = (rand()%(2240+1)-1120);
		float rho = rho_/10.0;
		// \sin\theta = [-1,1]
		int sin_theta_ = (rand()%(100+100+1)-100);
		float sin_theta = sin_theta_/100.0;
		cout << "rho is : " << rho << ";\t sin_theta is: " << sin_theta << endl;

		int noise_num = (rand()%(3000-1000+1)+1000);

		float cos_theta = sqrt(1- sin_theta*sin_theta);
		Point p1,p2;
		p1.x = -112;
		p2.x = 112;
		if ( abs(sin_theta)<1e-2 ){
			p1.y = p2.y = 112 - rho;
			p1.x = 0;
			p2.x = 224;
		}else{
			p1.y = -cos_theta*(1.0/sin_theta)*p1.x + rho*(1/sin_theta);
			p2.y = -cos_theta*(1.0/sin_theta)*p2.x + rho*(1/sin_theta);
			p1.x = 0;
			p2.x = 224;
			p1.y = 112 - p1.y;
			p2.y = 112 - p2.y;
		}

		line(image, p1, p2, Scalar(255), 3);
		cout << "p1 is : " << p1 << ";\t p2 is: " << p2 << endl;
		float d = sqrt(pow(p1.x-p2.x,2)+pow(p1.y-p2.y,2));
		if (d < 50 ){
			i--;
			continue;
		}

		for (size_t j=0; j<noise_num; ++j){
			int noise_x = (rand()%(224+1));
			int noise_y = (rand()%(224+1));
			circle(image, Point(noise_x,noise_y), 1,Scalar(255),-1); 
		}
		// cv::imshow("image", image);
		string name = imagesPath + to_string(i) + ".png";
		cv::imwrite(name, image);
		// waitKey(0);
		outfile << name << " " << rho/112.0 << " " << sin_theta << endl;
		
	}
	outfile.close();
	return 0;
}