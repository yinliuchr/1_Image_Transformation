#include "Preprocess.h"

int main() {
	Image img;
	img.readImg("car.jpg");
	if (!img.ifreadFail()) {
		img.showInput();
		char c = '1';
		while (c != 'q') {
			cout << endl << "Input the manipulation:"<< "'q' for exit£¬'1' for image rotation, '2' for image distorsion, '3' for TPS." << endl;
			c = _getch();
			while (c != 'q'&&c != '1'&&c != '2'&&c != '3') {
				cout << "Wrong input! Please try again:" << endl;
				c = _getch();
			}
			if (c == 'q') break;
			else if (c == '1') img.rotate();
			else if (c == '2') img.distorsion();
			else if (c == '3') img.TPS();
			img.afterProc();
			img.showInput();
		}
	}
	else cout << "Error in loading image, exit automatically."<<endl;
	
	return 0;
}