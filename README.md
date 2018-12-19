<!DOCTYPE html>
<html>
<body>
  <div id="readme" class="readme blob instapaper_body">
    <article class="markdown-body entry-content" itemprop="text"><h1><a id="user-content-improving-ir-based-bug-localization-with-context-aware-query-reformulation" class="anchor" aria-hidden="true" href="#improving-ir-based-bug-localization-with-context-aware-query-reformulation"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>Meta-Learning based Recommender System to Recommend Developers for Crowdsourcing Software Development</h1>
<h2><a id="user-content-accepted-paper-at-esecfse-2018" class="anchor" aria-hidden="true" href="#accepted-paper-at-esecfse-2018"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>Project for the submitted paper for Empirical Software Engineering Journal</h2>

<pre><code>
This is the project for our paper that proposed a meta-learning based recommender system to recommend reliable developers for crowdsourcing software development(CSD).
We shall give an insturction that will guide you to use the source code in this project here in detail.
</code></pre>


<h2>
<a id="user-content-subject-systems-6" class="anchor" aria-hidden="true" href="#subject-systems-6"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>
Instruction for building the recommender system from source code and executing experiments
</h2>
<ul>
<li>Prepare system environment</li>
<li>Start to run the data Crawler</li>
<li>Construct Input Data</li>
<li>Train Meta Models</li>
<li>Run Baselines and Policy Model for experiments</li>
</ul>

<h2><a id="user-content-materials-included" class="anchor" aria-hidden="true" href="#materials-included"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>
Prepare system environment
</h2>
<p><strong>Minimum configuration of machines</strong></p>

<ul>
<li><code>RAM:</code> 256G</li>
<li><code>CPU:</code> 12 logic cores</li>
<li><code>Disk:</code> 1TB+</li>
<li>TitanXP NVIDIA GPU is recommended for boosting computation</li>
<li>Make sure the bandwidth is at least 1000Mb/s if the database is not in your programming machine</li>

</ul>
<p><strong>Install python environment</strong></p>
<p>We develop the whole system using python, so we recommend you to install an anaconda virtual python3.6 environment at: https://www.anaconda.com/
</p>

<p><strong>Install Mysql Database</strong></p>
<p>
Install mysql database into your computer with a linux system, and configure mysql ip and port according to the instruction of https://www.mysql.com/.
</p>

<p><strong>Install JDK8 and relative JAVA runtime</strong></p>
<p>
We use the crawler program implemented in JAVA. 
Please refer to the topcoder project at: https://github.com/lifeloner/topcoder for newest data crawler implemented in JAVA and prepare to import relative jar libraries. 
</p>

<p><strong>Required python packages</strong></p>
<ul>
<li><code>machine learning:</code>scikit-learn, lightgbm, xgboost, tensorflow, keras, imbalance-learn, networkx</li>
<li><code>data preprocessing:</code> pymysql, numpy, pandas</li>
<li><code>models:</code> Models required for the tool</li>
</ul>

<p><strong>Project Check</strong></p>
<ul>
<li>The DIG is implemented in CompetitionGraph Package. </li>
<li>The machine learning algorithms and policy model are implemented in ML_Models package. </li>
<li>For challenge and developer feature encoding and some data preprocessing modules of the system, refer to the DataPre package. </li>
<li>The Utility package contains some personalized tag definition, user function and testing scripts.</li>
<li> Make sure that the hierarchy of data folder is same in local disk. </li>
</ul>


<h2><a id="user-content-available-operations" class="anchor" aria-hidden="true" href="#available-operations"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>
Start to run the data Crawler
</h2>
<p>We do have a database in our laboratory, but due to the size and continuously updating of our database, it is not a good way to put the database here. 
Instead, we put the tools for data collection here, thus everyone can get enough data as they want. 
If you are eager for our data, contact me via the anonymous email mail@{1196641807@qq.com}. 
</p>
<ul>
<li>
Install mysql database into your computer with a linux system, and configure mysql ip and port according to the instruction of https://www.mysql.com/.
</li>
<li>
refer to the topcoder project at: https://github.com/lifeloner/topcoder for newest data crawler implemented in JAVA. 
</li>
<li>
After downloading the java crawler maven project, please use intelliJ idea at: https://www.jetbrains.com/idea/ to deploy the crawler jar package in your machine
</li>
<li>
Configure the ip and port of your crawler according to the the configure of mysql database
</li>
<li>
Start run the crawler by the following command which will run in background: <br\>
nohup java –jar crawler.jar &amp;
</li>
</ul>

<h2><a id="user-content-required-parameters-for-the-operations" class="anchor" aria-hidden="true" href="#required-parameters-for-the-operations"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>
Construct Input Data
</h2>
<p><strong>Configure the datra/dbSetup.xml and set ip and port as same as the machine running mysql database, 
copy data/viewdef.sql and run it in your mysql client to create view for initial data cleaning.</strong></p>


<p><strong>You need to encode Developer and Challenge features at first</strong></p>
  <ul>
  <li>
  Run TaskContent.py of DataPre package to generate challenge feature encoding vectors and build clustering model
  </li>
  <li>Run UserHistory.py of DataPre package to generate developer history data
  </li>
  <li>Run DIG.py of CompetitionGraph package to generate developer rank score data
  </li>
  </ul>

<p><strong>Run TaskUserInstances.py of DataPre package to generate input data</strong></p>
  <ul>
  <li>Adjust the maxProcessNum of DataInstances class to adapt your computer CPU and RAM
  </li>
  <li>For training,set global variant testInst=False. The value of variant mode in global means 0-registration training data input, 1-submission training data input, 2-winning training data input. You have to run the script under the 3 values.
  </li>
  <li>Generate test input data via set mode=2 and testinst=True
  </li>
  </ul>

<p><strong>After finished running all the above scripts, check whether the generate traing input and test input data is completed via running the TopcoderDataset.py</strong></p>

<h2><a id="user-content-q1-how-to-install-the-blizzard-tool" class="anchor" aria-hidden="true" href="#q1-how-to-install-the-blizzard-tool"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>
Train Meta Models
</h2>
<p><strong>Run XGBoostModel.py of ML_Models package</strong></p>
<ul>
<li>Feed “keepd” as key of tasktypes and run the script for 3 times with mode =0,1,and 2
</li>
<li>Feed “clustered” as key of tasktrypes and run the script for 3 times with mode=0,1,and 2
</li>
<li>After finished this, the meta model implemented using XGBoost algorithms can extract registration meta-feature, submission meta-feature and winning met-feature of all datasets
</li>
</ul>
<p><strong>Run DNNModel.py of ML_Models package in the same way as XGBoostModel.py
</strong></p>
<p><strong>Run EnsembleModel.py of ML_Models package in the same way as XGBoostModel.py
</strong></p>
<p><strong>Generate the performance of all the winning meta models via running MetaModelTest.py of ML_Models package
</strong></p>
<ul>
<li>Readers can build winning predictor based on the performance results
</li>
</ul>

<h2><a id="user-content-query-file-format" class="anchor" aria-hidden="true" href="#query-file-format"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>
Run Baselines and Policy Model for experiments
</h2>
<p><strong>Run BaselineModel.py of ML_Models package to build the baseline models we mentioned in the paper
</strong></p>
<ul>
<li>
After building baseline models, run the MetaModelTest.py of ML_Models package again but pass the model name as the names of classes of the baseline model in BaselineModel.py to generate performance results
</li>
</ul>
<p><strong></strong></p>
<ul>
<li>
Readers can refer to MetaLearning.py of ML_Models package which implemented some new learning process but may not be global optima
</li>
</ul>
<p>..........................................................</p>



<h2><a id="user-content-please-cite-our-work-as" class="anchor" aria-hidden="true" href="#please-cite-our-work-as"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>
Please give a cite to our work if you want use the project somewhere else. 
</h2>
<pre><code>@INPROCEEDINGS{metalearning-recommender, 
author={Zhenyu Zhang, Hailong Sun, HongyuZhang}, 
title={Developer Recommendation for Topcoder through aMeta-learning based Policy Model},
year={2019},
url={https://github.com/zhangzhenyu13/CSDMetalearningRS} 
}

  </body>
</html>

