#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

int main()
{
    //两幅图中粗提取的特征点
    std::vector<KeyPoint> keypoints_1, keypoints_2;
    //两幅图中筛选后的特征点
    std::vector<KeyPoint> good_keypoints_1, good_keypoints_2;
    //两幅图对应的描述矩阵
    Mat descriptors_1, descriptors_2;
    //匹配相关
    BFMatcher matcher(cv::NORM_HAMMING);
    std::vector< DMatch > matches;
    std::vector<DMatch> good_matches;
    double max_dist = 0; double min_dist = 100;
    std::vector<Point2f> frame1_features_ok, frame2_features_ok;
    //临时变量
    int i, count = 0;
    Point2f tmpPoint;
    //拼接后的图像
    Mat dstImage;
    //匹配展示图像
    Mat img_goodmatch;
    //匹配的坐标
    ofstream SaveMatches("matches.txt");
    
    //第一步，读取图像
    Mat src1 = imread("src1.jpg");
    Mat src2 = imread("src2.jpg");
  
    //第二步，创建ORB特征，并分别在两幅图中提取ORB特征
    Ptr<ORB> orb = ORB::create(500,1.2f,8,31,0,2,ORB::HARRIS_SCORE,31,20);
    orb->detectAndCompute(src1,Mat(),keypoints_1,descriptors_1);
    orb->detectAndCompute(src2,Mat(),keypoints_2,descriptors_2);

    //第三步，暴力匹配
    matcher.match(descriptors_1, descriptors_2, matches);  
    
    //第四步，获取匹配点对间的最大最小距离
    for( int i = 0; i < descriptors_1.rows; i++ ){  
        double dist = matches[i].distance;  
        if(dist < min_dist )   
            min_dist = dist;  
        if(dist > max_dist )  
            max_dist = dist;   
    }    
    
    //第五步，筛选好的匹配特征
    for(i = 0; i < descriptors_1.rows; i++ ){  
        if(matches[i].distance <= max(2*min_dist,30.0)){    
            good_matches.push_back(matches[i]);  
            good_keypoints_1.push_back(keypoints_1[matches[i].queryIdx]);  
            good_keypoints_2.push_back(keypoints_2[matches[i].trainIdx]);  
        }  
    }   
    
    //第六步，提取好特征点的坐标
    for(int i=0;i<good_keypoints_1.size();i++){
      
      tmpPoint.x = good_keypoints_1[i].pt.x;
      tmpPoint.y = good_keypoints_1[i].pt.y;
      frame1_features_ok.push_back(tmpPoint);
       
      tmpPoint.x = good_keypoints_2[i].pt.x;
      tmpPoint.y = good_keypoints_2[i].pt.y;
      frame2_features_ok.push_back(tmpPoint);
    }
    
    //第七步，计算H矩阵
    Mat homo=findHomography(frame2_features_ok,frame1_features_ok,CV_RANSAC);
    cout<<homo<<endl;
    
    //第八步，图像配准,需制定输出图像大小，这里设置为原图的1.2倍，将src2的内容按照H矩阵重采样到src1的范围中
    warpPerspective(src2, dstImage, homo, Size(int(src2.cols*1.2),int(src2.rows*1.2)));
    
    //第九步，将src1的内容复制到dstImage中
    src1.copyTo(dstImage(Rect(0, 0, src1.cols, src1.rows)));
    
    //第十步，绘制匹配的点
    drawMatches ( src1, keypoints_1, src2, keypoints_2, good_matches, img_goodmatch );
    
    //第十一步，展示成果
    imshow("匹配图像", dstImage);
    imshow ( "优化后匹配点对", img_goodmatch );
    waitKey(0);
    
    //第十二步，输出影像
    imwrite("dst.jpg",dstImage);
    imwrite("match.jpg",img_goodmatch);
    
    //第十三步，输出匹配的坐标
    for(int i=0;i<good_keypoints_1.size();i++)
    {
      SaveMatches<<frame1_features_ok[i].x<<","<<frame1_features_ok[i].y<<" "<<frame2_features_ok[i].x<<","<<frame2_features_ok[i].y<<endl;
    }
    SaveMatches.close();
    
    return 0;
}