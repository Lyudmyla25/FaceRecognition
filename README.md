# FaceRecognition

Project uses two different ideas on face recognition. Database has 102 people each two pictures. 

1. First using neural networking to generate 1000 points vector and determine 1000 properties in each photo, 
this data then analised by me using random forests. But result wasn't as good as I expected. (on master branch)

2. Second approach using neural networking to generate only 5 points on face (eyes, ears, nose) then we calculated all distances 
between them (10 pairs), skale them to distance between eyes and run logistic regresion on them. 
Score is around 70%. Where this is confusion table [[175, 107],[ 64, 254]]. (on biometric branch)

Final version is version on biometric branch.
