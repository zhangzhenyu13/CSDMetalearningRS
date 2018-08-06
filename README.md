# CSDMetalearningRS
a meta learning based CSD recommender

The rest of the README is a brief decription for how to use the project, for more detailed information please refer to the SystemInstruction.pdf of our rpoject.





The DIG is implemented in CompetitionGraph Package.
The machine learning algorithms and policy model are implemented in ML_Models package. 
For challenge and developer feature encoding and some data preprocessing modules of the system, refer to the DataPre package. 
The Utility package contains some personalized tag definition, user fucntion and testing scripts. 

Before running the system, make sure to configure following settings:
install python 3.x;
pip install keras, tensorflow, scikit-learn, imbalance-learn, numpy. pymysql;
install mysql database;
refer to the topcoder project at: https://github.com/lifeloner/topcoder for newest data crawler implemented in JAVA.
customize local mysql database ip and port accroding to local machine configuration.

Make sure that the hierarchy of data folder is same in local disk.
Run DataPre package script to start Data Extractor and generate input data for the system.
Run ML_Models package script to start meta-model training and optimal meta feature searching.
Test Policy Model.
