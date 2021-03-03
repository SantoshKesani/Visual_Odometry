import numpy as np
import cv2
import glob
import random
import matplotlib.pyplot as plt
from skimage.feature import match_descriptors, ORB, plot_matches
from operator import itemgetter
import math

from ReadCameraModel import *
##import ReadCameraModel.py
from UndistortImage import *
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D


def feature_detector_orb(img1, img2):
    print("Running Feature Detector")
    # Initiating the ORB detector
##    descriptor_extractor = ORB()
    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    print("Completed Detect and Compute")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)

    img_match = cv2.drawMatches(img1,kp1,img2,kp2,matches[:20],None, flags=2)
##    cv2.imshow("Point Matches", img_match)
##    cv2.waitKey(0)

    desc_match = match_descriptors(des1, des2, cross_check=True)

##    print("Reduced Matches")
##    print(desc_match[:20])

    pts1 = []
    pts2 = []

    for mat in matches:
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        pts1.append((x1,y1))
        pts2.append((x2,y2))

##    print("Transfered Points")
##    print(pts1)
##    print(pts2)

    pts_list = [pts1, pts2]
    
    return pts_list

def Fundamental_matrix(pc1, pc2):
    n = len(pc1)
    # if pc2.shape[1] != n:
    #     raise ValueError("Number of points don't match.")

    A = np.zeros((n, 9))
    for i in range(n):
        A[i] = np.array([pc2[i][0]*pc1[i][0], pc2[i][0]*pc1[i][1], pc2[i][0], pc2[i][1]*pc1[i][0], pc2[i][1]*pc1[i][1],
                        pc2[i][1], pc1[i][0], pc1[i][1], 1])

    # Constraint Enforcement - SVD decomposition
    [U, D, Vt] = np.linalg.svd(A)
    L = Vt[-1, :]
    F = L.reshape(3, 3)

    # Rank Enforcement(Improvising F)
    [u, d, vt] = np.linalg.svd(F)
    d[2] = 0
    F_new = np.multiply(u, np.multiply(np.diag(d), vt))
    # print(F_new)

    return F_new


def camera_poses(essential_matrix):
    U, D, Vt = np.linalg.svd(essential_matrix, full_matrices=True)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    C1 = U[:, 2]
    C2 = -U[:, 2]
    C3 = U[:, 2]
    C4 = -U[:, 2]
    C_list = [C1, C2, C3, C4]

    R1 = np.dot(U, np.dot(W, Vt))
    R2 = np.dot(U, np.dot(W, Vt))
    R3 = np.dot(U, np.dot(W.T, Vt))
    R4 = np.dot(U, np.dot(W.T, Vt))
    
    R_list = [R1, R2, R3, R4]

    for i in range(0, len(C_list)):
        det_R = np.linalg.det(R_list[i])
        if det_R < 0:
            C_list[i] = -C_list[i]
            R_list[i] = -R_list[i]

    return C_list, R_list


def ransac(all_points,imgP1,imgP2):
    maxiter = 8000
##    maxiter = 5000
##    threshold = 1e-6
    threshold = 1e-3
    s1 = []
    s2 = []
    outliers = []
    n = 0
    inliers_1 = []
    inliers_2 = []

    for i in range(maxiter):
##    for i in range(1):
##        rand_points = random.sample(range(0, len(all_points[0])), 8)
        rand_points = random.sample(range(0, len(imgP1)), 8)

        pc1 = []
        pc2 = []
        
        for k in rand_points:
##            print(k)
            pc1.append(imgP1[k])
            pc2.append(imgP2[k])
            
##        pc1 = list(itemgetter(*rand_points)(all_points[0]))
##        pc2 = list(itemgetter(*rand_points)(all_points[1]))

##        print("PC1")
##        print(pc1)
##        print("PC2")
##        print(pc2)
        
        F = Fundamental_matrix(pc1, pc2)

##        print("Fund Mat")
##        print(F)

        for j in range(7):
            pc1_3d = np.array([[pc1[j][0]], [pc1[j][1]], [1]])
            pc2_3d = np.array([[pc2[j][0]], [pc2[j][1]], [1]])

##            print("PC1 3D")
##            print(pc1_3d)
##            print("PC2 3D")
##            print(pc2_3d)

            pc2_3d_T = np.transpose(pc2_3d)
          
            if abs(np.dot(pc2_3d_T,np.dot(F,pc1_3d))) < threshold:
##            if abs(np.matmul(np.transpose(pc2_3d), np.matmul(F, pc1_3d))) < threshold:
                # print(abs(np.matmul(np.transpose(x_i_dash[j]), np.matmul(F_tilde, x_i[j]))))
                s1.append(pc1[j])
                s2.append(pc2[j])
                # print(s)
            else:
                outliers.append([pc1[j], pc2[j]])

        if n < len(s1):
            n = len(s2)
            inliers_1 = s1
            inliers_2 = s2

##    print(inliers_1)
##    print(inliers_2)
##    print(outliers)
    
    return [inliers_1, inliers_2]


def Essential_matrix(fundamental_matrix, calibration_matrix):
    calibration_matrix_T = np.transpose(calibration_matrix)
    essential_matrix = np.dot(calibration_matrix_T,np.dot(fundamental_matrix,calibration_matrix))
    
##    essential_matrix = np.matmul(np.transpose(calibration_matrix), np.matmul(fundamental_matrix, calibration_matrix))

    # Enforce rank two:
##    [u, d, vt] = np.linalg.svd(essential_matrix)
##
##    d[2]=0
##    essential_matrix = np.multiply(u, np.multiply(np.diag(d), vt))

    return essential_matrix

def linear_triangulation(P1, P2, pt1, pt2):
    """point are in shape (2,1)"""
    
    row_1 = (pt1[0] * P1[2, :] - P1[0, :]).reshape(1, 4)
    row_2 = (-pt1[1] * P1[2, :] + P1[1, :]).reshape(1, 4)
    row_3 = (pt2[0] * P2[2, :] - P2[0, :]).reshape(1, 4)
    row_4 = (-pt2[1] * P2[2, :] + P2[1, :]).reshape(1, 4)
    
    A_mat = np.concatenate((row_1, row_2, row_3, row_4), axis=0)
    
    U, D, Vt = np.linalg.svd(A_mat)
    
    X = Vt[-1, :]
    X = X/X[3]
    X = X.reshape(4, 1)
    X = X[0:3].reshape(3, 1)

    return X


def disambugate_camerapose(R_matrix, C_matrix, K, ft1, ft2):
    choices = np.zeros(4).astype(int)
    I = np.eye(4)
    Rt_1 = I[0:3, :]
    P1 = np.dot(K, Rt_1)
    P2 = np.zeros((3,4))

##    print("R Matrix Length")
##    print(len(R_matrix))    
    for i in range(len(R_matrix)):
        count = 0
        Cmat_T = np.transpose(C_matrix[i])
        Rmat = R_matrix[i]

        for row in range(len(P2)):
            for col in range(len(P2[0])-1):
                P2[row][col] = Rmat[row][col]
            P2[row][col+1] = Cmat_T[row]

        for j in range(len(ft1)):
            X = linear_triangulation(P1, P2, ft1[j], ft2[j])
            r3 = R_matrix[i][2, :].reshape((1, 3))

            if np.dot(r3, np.subtract(X,P2[2, 3])) > 0:
##            if np.dot(r3, (X-C)) > 0:
                count += 1
        choices[i] = count
    ind = choices.argmax()

    return C_matrix[ind], R_matrix[ind]

def basic_process_image(img):

    color_image = cv2.cvtColor(img, cv2.COLOR_BayerGR2BGR)
    undistorted_image = UndistortImage(color_image, LUT)
    gray_image = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)
    # Hist_equ_img = cv2.equalizeHist(gray_image)
    # final_image = cv2.GaussianBlur(Hist_equ_img, (3, 3), 0)

    return gray_image

def display_points(img_pts1, img_pt2, image1, image2):

    imgP1 = []
    imgP2 = []

    for (x,y) in img_pts1:
        x1 = int(x)
        y1 = int(y)
        imgP1.append((x1,y1))

    for (x,y) in img_pts2:
        x2 = int(x)
        y2 = int(y)
        imgP2.append((x2,y2))

##    print("Integeter List of Points:")
##    print(imgP1)
##    print(imgP2)

##    cv2.namedWindow("Image 1 Points")
##    for (x,y) in imgP1:
##        cv2.circle(image1,(x,y),3,(0,0,255),-1)
##    cv2.imshow("Image 1 Points",image1)

##    cv2.namedWindow("Image 2 Points")
##    for (x,y) in imgP2:
##        cv2.circle(image2, (x,y),2,(0,0,255), -1)
##    cv2.imshow("Image 2 Points",image2)

##    cv2.waitKey(0)

    return

def plot_camera_pose(T,R,Pos,Ang,count):
    New_Pos = [0,0,0]
    New_Ang = [0,0,0]

    length = 10

    # Recalculate Translation
##    for i in range(len(Pos)):
##        New_Pos[i] = Pos[i]+T[i]

    # Calculate New Pose Locations for X, Y, Z directions
    New_Pos[0] = Pos[0]+T[0]
    New_Pos[1] = Pos[1]+T[1]
    New_Pos[2] = Pos[2]+T[2]

    ThetX = np.arctan2(R[2][1],R[2][2])
    calc_temp = math.sqrt((R[2][1])**2+(R[2][2])**2)
    ThetY = np.arctan2(-R[0][2],calc_temp)
    ThetZ = np.arctan2(R[0][1],R[0][0])

    Thet_Calc = [ThetX,ThetY,ThetZ]

##    print("Theta Calc")
##    print(Thet_Calc)

##    for k in range(len(Ang)):
##        New_Ang[k] = Ang[k]+Thet_Calc[k]

    # Calculate New Pose Angles for X, Y, Z axis
    New_Ang[0] = Ang[0]+Thet_Calc[0]
    New_Ang[1] = Ang[1]+Thet_Calc[1]
    New_Ang[2] = Ang[2]+Thet_Calc[2]

    print("New Angles")
    print(New_Ang)     
    
    # Calculate end points of End angles.
##    end_ptsX = New_Pos[0] + (length*math.sin(New_Ang[0]))
##    end_ptsY = New_Pos[1] + (length*math.sin(New_Ang[1]))
##    end_ptsZ = New_Pos[2] + (length*math.sin(New_Ang[2]))

    end_ptsX = New_Pos[0] + (length*math.sin(New_Ang[1])*math.cos(New_Ang[2]))
    end_ptsY = New_Pos[1] + (length*math.sin(New_Ang[2]))
    end_ptsZ = New_Pos[2] + (length*math.cos(New_Ang[2])*math.cos(New_Ang[1]))

##    end_ptsX = New_Pos[0] + (length*math.sin(Thet_Calc[0]))
##    end_ptsY = New_Pos[1] + (length*math.sin(Thet_Calc[1]))
##    end_ptsZ = New_Pos[2] + (length*math.sin(Thet_Calc[2]))

    if end_ptsY > 0:
        end_ptsX = -end_ptsX
        end_ptsY = -end_ptsY
        end_ptsZ = -end_ptsZ
    
    end_pts = [end_ptsX,end_ptsY,end_ptsZ]

    print("New Pts")
    print(New_Pos)

##    print("End Pts")
##    print(end_pts)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.set_xlim([-20,20])
    ax1.set_ylim([20,-20])
    ax1.set_zlim([-20,20]) 

    ax1.set_xlabel('X')
    ax1.set_ylabel('Z')
    ax1.set_zlabel('Y')
    ax1.set_title('Camera Pose')

    # Unhide for Horizontal Display Orientation
##    ax1.plot([New_Pos[0],end_pts[0]],[New_Pos[2],end_pts[2]],zs=[New_Pos[1],end_pts[1]])
##    ax1.scatter(New_Pos[0],New_Pos[2],zs=[New_Pos[1]],c='green')
##    ax1.scatter(end_pts[0],end_pts[2],zs=[end_pts[1]],c='red')

    # Unhide for Vertical Display Orientation
    ax1.plot([New_Pos[0],end_pts[0]],[New_Pos[1],end_pts[1]],zs=[New_Pos[2],end_pts[2]])
    ax1.scatter(New_Pos[0],New_Pos[1],zs=[New_Pos[2]],c='green')
    ax1.scatter(end_pts[0],end_pts[1],zs=[end_pts[2]],c='red')

##    plt.show()

    plt.savefig("Pose_Plots/Pose{:06}.png".format(count))

    plt.close()

    fig2 = plt.figure()
    ax2 = plt.axes()
    ax2.set_xlim([-20,20])
    ax2.set_ylim([5,-20])

    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.set_title('2D Camera Pose')

    ax2.plot([New_Pos[0],end_pts[0]],[New_Pos[1],end_pts[1]])
    ax2.scatter(New_Pos[0],New_Pos[1],c='green')
    ax2.scatter(end_pts[0],end_pts[1],c='red')

##    plt.show()

    plt.savefig("Pose_Plots/2D_Pose{:06}.png".format(count))

    plt.close()

    return [New_Pos,New_Ang,end_pts]
##    Axes3D.plot()

def Comparison(points0,points1, K):

    Comp_Pos = [0,0,0]
    Comp_Ang = [0,0,0]

    trace_length = 10
    
    points0 = np.array(points0)
    points1 = np.array(points1)

    E_comp, mask = cv2.findEssentialMat(points0,points1,K,method = 0,prob = 0.98,threshold = 1.0)

    _,R,T,_ = cv2.recoverPose(E_comp,points0,points1,K)

    Tmat = []
    for l in range(len(T)):
        Tmat.append(T[l][0])

##    print("Tmat")
##    print(Tmat)
    
    Comp_Pos[0] = Comp_Pos[0]+Tmat[0]
    Comp_Pos[1] = Comp_Pos[1]+Tmat[1]
    Comp_Pos[2] = Comp_Pos[2]+Tmat[2]

    print("Comparison Position")
    print(Comp_Pos)

    Comp_ThetX = np.arctan2(R[2][1],R[2][2])
    calc_temp = math.sqrt((R[2][1])**2+(R[2][2])**2)
    Comp_ThetY = np.arctan2(-R[0][2],calc_temp)
    Comp_ThetZ = np.arctan2(R[0][1],R[0][0])

    Thet_Comp = [Comp_ThetX,Comp_ThetY,Comp_ThetZ]

##    for k in range(len(Ang)):
##        New_Ang[k] = Ang[k]+Thet_Calc[k]

    # Calculate New Pose Angles for X, Y, Z axis
    Comp_Ang[0] = Comp_Ang[0]+Thet_Comp[0]
    Comp_Ang[1] = Comp_Ang[1]+Thet_Comp[1]
    Comp_Ang[2] = Comp_Ang[2]+Thet_Comp[2]

    print("Comparison Angles")
    print(Comp_Ang)     
    
    # Calculate end points of End angles.
    Comp_endX = Comp_Pos[0] + (trace_length*math.sin(Comp_Ang[1])*math.cos(Comp_Ang[2]))
    Comp_endY = Comp_Pos[1] + (trace_length*math.sin(Comp_Ang[2]))
    Comp_endZ = Comp_Pos[2] + (trace_length*math.cos(Comp_Ang[2])*math.cos(Comp_Ang[1]))

    if Comp_endY > 0:
        Comp_endX = -Comp_endX
        Comp_endY = -Comp_endY
        Comp_endZ = -Comp_endZ

    Comp_endpts = [Comp_endX, Comp_endY, Comp_endZ]

    return [E_comp,R,T,Comp_Pos,Comp_Ang,Comp_endpts]

def Drift_Comparison(New_Pos, New_Ang, Comp_Pos, Comp_Ang, end_pts, Comp_endpts, count):
    Pos_Drift = []
    Ang_Drift = []

    CompPts = []
    CompEndPts = []

    for k in range(len(Comp_Pos)):
        CompPts.append(Comp_Pos[k])
        CompEndPts.append(Comp_endpts[k])

    for i in range(len(New_Pos)):
        Pos_Drift.append(New_Pos[i]-Comp_Pos[i])
        Ang_Drift.append(New_Ang[i]-Comp_Ang[i])
    
    print("Position Drift")
    print(Pos_Drift)
    print("Angular Drift")
    print(Ang_Drift)
    print("\n")

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, projection='3d')
    ax3.set_xlim([-20,20])
    ax3.set_ylim([20,-20])
    ax3.set_zlim([-20,20]) 

    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_zlabel('Y')
    ax3.set_title('Function Comparison')

    # Unhide for Horizontal Display Orientation
    ax3.plot([New_Pos[0],end_pts[0]],[New_Pos[1],end_pts[1]],zs=[New_Pos[2],end_pts[2]])
    ax3.scatter(New_Pos[0],New_Pos[1],zs=[New_Pos[2]],c='green')
    ax3.scatter(end_pts[0],end_pts[1],zs=[end_pts[2]],c='red')

    ax3.plot([CompPts[0],CompEndPts[0]],[CompPts[1],CompEndPts[1]],zs=[CompPts[2],CompEndPts[2]])
    ax3.scatter(CompPts[0],CompPts[1],zs=[CompPts[2]],c='black')
    ax3.scatter(CompEndPts[0],CompEndPts[1],zs=[CompEndPts[2]],c='gray')

##    plt.show()

    plt.savefig("Pose_Plots/Pose_Comparison{:06}.png".format(count))

    plt.close()

if __name__ == "__main__":
    # Reading Parameters
    fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel('Dataset/model')
    # Intrinsic Parameters Matrix
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

##    Pos = [0, 0, 0]
##    Ang = [(np.pi/2),(np.pi/2),0]
    # Importing The images
    bayer_images = [img for img in glob.glob("Dataset/stereo/centre/*.png")]
    bayer_images.sort()

    frame_number = 0
    scan_frames = len(bayer_images)

    pose_frames = []

    for s in range(scan_frames-1):

        Pos = [0, 0, 0]
##        Ang = [(np.pi/2),(np.pi/2),0]
        Ang = [0, 0, 0]

        # Basic Processing
        image1 = cv2.imread(bayer_images[frame_number], 0)
        new_image1 = basic_process_image(image1)
        image2 = cv2.imread(bayer_images[frame_number+1], 0)
        new_image2 = basic_process_image(image2)


##        print("K Matrix")
##        print(K)
        
        '''Step - 1'''
        # Key-point Algorithm:
        p = feature_detector_orb(new_image1, new_image2)

        img_pts1 = p[0]
        img_pts2 = p[1]

        display_points(img_pts1, img_pts2, image1, image2)

        imgP1 = []
        for (x,y) in img_pts1:
            x1 = int(x)
            y1 = int(y)
            imgP1.append((x1,y1))

        imgP2 = []
        for (x,y) in img_pts2:
            x2 = int(x)
            y2 = int(y)
            imgP2.append((x2,y2))
     
        '''Step - 2'''
        # Inliers through ransac:
        inliers = ransac(p,imgP1,imgP2)

        # Fundamental Matrix:
        Fund_mat = Fundamental_matrix(inliers[0], inliers[1])

    ##    print("Fundamental Matrix")
    ##    print(Fund_mat)

        '''Step - 3'''
        # Essential Matrix:
        E = Essential_matrix(Fund_mat, K)

        print("Fundamental Matrix")
        print(Fund_mat)
        print("Essential Matrix: ")
        print(E)
        
        '''Step - 4'''
        # Camera poses:
        C, R = camera_poses(E)
##        print("C Matrix: ")
##        print(C)
##        print("R Matrix: ")
##        print(R)
        
        '''Step - 5'''
        # Recovering the correct C, R:
        T_final, R_final = disambugate_camerapose(R, C, K, inliers[0], inliers[1])

##        print("T Final")
##        print(T_final)
##        print("R Final")
##        print(R_final)
        
        '''Step - 6'''
        # Plotting the camera centers:
        New_Pos,New_Ang,end_pts = plot_camera_pose(T_final, R_final, Pos, Ang, s)

##        Pos = New_Pos
##        Ang = New_Ang

        '''Step - 7'''
        # Extra Credit: Comparison
        E_comp,R_comp,T_comp,Comp_Pos,Comp_Ang,Comp_endpts = Comparison(imgP1,imgP2, K)

        Drift_Comparison(New_Pos, New_Ang, Comp_Pos, Comp_Ang, end_pts, Comp_endpts, s)

        

        frame_number = frame_number+1

##        print("Essential Matrix: ")
##        print(E)
##
##        print("E Comparison Matrix")
##        print(E_comp)
##
##        print("T Comparison Matrix")
##        print(T_comp)
##
##        print("R Comparison Matrix")
##        print(R_comp)


'''code for non-linear triangulation'''
def nonlinear_triangulation(R_final, C_final, K, ft1, ft2):
    I = np.eye(4)
    Rt_1 = I[0:3, :]
    P1 = np.dot(K, Rt_1)
    P2 = np.vstack(R_final,C_final,axis=1)
    test = []

    for j in range(len(ft1)):
        p1_3d = np.array([[ft1[j][0]], [ft1[j][1]], [1]])
        p2_3d = np.array([[ft2[j][0]], [ft2[j][1]], [1]])

        X = linear_triangulation(P1, P2, ft1[j], ft2[j])
        X_homo = np.array([X[0][0],X[1][0],X[2][0],1]).reshape(4,1)
        A = np.linalg.norm(np.multiply(P1, X)-p1_3d)
        B = np.linalg.norm(np.multiply(P2, X)-p2_3d)
        f_min = A**2 + B**2
        test.append(f_min))
    test = np.asarray(test)
    ind = test.argmax()
    X_final = linear_triangulation(P1, P2, ft1[ind], ft2[ind])

    return X_final
