<!DOCTYPE html>
<html>
<body>
  <div id="readme" class="readme blob instapaper_body">
    <article class="markdown-body entry-content" itemprop="text"><h1><a id="user-content-improving-ir-based-bug-localization-with-context-aware-query-reformulation" class="anchor" aria-hidden="true" href="#improving-ir-based-bug-localization-with-context-aware-query-reformulation"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>Meta-Learning based Recommender System to Recommend Developers for Crowdsourcing Software Development</h1>
<h2><a id="user-content-accepted-paper-at-esecfse-2018" class="anchor" aria-hidden="true" href="#accepted-paper-at-esecfse-2018"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>Project for the submitted paper for ICSE 2019</h2>
<pre><code>

</code></pre>

<pre><code>
This is the project for our paper that proposed a meta-learning based recommender system to reconmmend reliable developers for crowdsourcing software development(CSD).
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

<p><strong>Total Bug reports: 5,139</strong></p>
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



<h2><a id="user-content-please-cite-our-work-as" class="anchor" aria-hidden="true" href="#please-cite-our-work-as"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>Please give cite our work if you want use the project somewhere else</h2>
<pre><code>@INPROCEEDINGS{metalearning-recommender, 
author={AnonymousAuthor2013}, 
title={Developer Recommendation for Crowdsourcing Software Development through a Meta-learning based Policy Model},
year={2019},
url={https://github.com/AnonymousAuthor2013/CSDMetalearningRS} 
}
</code></pre>
</article>
  </div>

  </div>

  <details class="details-reset details-overlay details-overlay-dark">
    <summary data-hotkey="l" aria-label="Jump to line"></summary>
    <details-dialog class="Box Box--overlay d-flex flex-column anim-fade-in fast linejump" aria-label="Jump to line">
      <!-- '"` --><!-- </textarea></xmp> --></option></form><form class="js-jump-to-line-form Box-body d-flex" action="" accept-charset="UTF-8" method="get"><input name="utf8" type="hidden" value="&#x2713;" />
        <input class="form-control flex-auto mr-3 linejump-input js-jump-to-line-field" type="text" placeholder="Jump to line&hellip;" aria-label="Jump to line" autofocus>
        <button type="submit" class="btn" data-close-dialog>Go</button>
</form>    </details-dialog>
  </details>


  </div>
  <div class="modal-backdrop js-touch-events"></div>
</div>

    </div>
  </div>

  </div>

        
<div class="footer container-lg px-3" role="contentinfo">
  <div class="position-relative d-flex flex-justify-between pt-6 pb-2 mt-6 f6 text-gray border-top border-gray-light ">
    <ul class="list-style-none d-flex flex-wrap ">
      <li class="mr-3">&copy; 2018 <span title="0.32219s from unicorn-585fc897bb-dkjgp">GitHub</span>, Inc.</li>
        <li class="mr-3"><a data-ga-click="Footer, go to terms, text:terms" href="https://github.com/site/terms">Terms</a></li>
        <li class="mr-3"><a data-ga-click="Footer, go to privacy, text:privacy" href="https://github.com/site/privacy">Privacy</a></li>
        <li class="mr-3"><a href="https://help.github.com/articles/github-security/" data-ga-click="Footer, go to security, text:security">Security</a></li>
        <li class="mr-3"><a href="https://status.github.com/" data-ga-click="Footer, go to status, text:status">Status</a></li>
        <li><a data-ga-click="Footer, go to help, text:help" href="https://help.github.com">Help</a></li>
    </ul>

    <a aria-label="Homepage" title="GitHub" class="footer-octicon mr-lg-4" href="https://github.com">
      <svg height="24" class="octicon octicon-mark-github" viewBox="0 0 16 16" version="1.1" width="24" aria-hidden="true"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>
</a>
   <ul class="list-style-none d-flex flex-wrap ">
        <li class="mr-3"><a data-ga-click="Footer, go to contact, text:contact" href="https://github.com/contact">Contact GitHub</a></li>
        <li class="mr-3"><a href="https://github.com/pricing" data-ga-click="Footer, go to Pricing, text:Pricing">Pricing</a></li>
      <li class="mr-3"><a href="https://developer.github.com" data-ga-click="Footer, go to api, text:api">API</a></li>
      <li class="mr-3"><a href="https://training.github.com" data-ga-click="Footer, go to training, text:training">Training</a></li>
        <li class="mr-3"><a href="https://blog.github.com" data-ga-click="Footer, go to blog, text:blog">Blog</a></li>
        <li><a data-ga-click="Footer, go to about, text:about" href="https://github.com/about">About</a></li>

    </ul>
  </div>
  <div class="d-flex flex-justify-center pb-6">
    <span class="f6 text-gray-light"></span>
  </div>
</div>



  <div id="ajax-error-message" class="ajax-error-message flash flash-error">
    <svg class="octicon octicon-alert" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M8.893 1.5c-.183-.31-.52-.5-.887-.5s-.703.19-.886.5L.138 13.499a.98.98 0 0 0 0 1.001c.193.31.53.501.886.501h13.964c.367 0 .704-.19.877-.5a1.03 1.03 0 0 0 .01-1.002L8.893 1.5zm.133 11.497H6.987v-2.003h2.039v2.003zm0-3.004H6.987V5.987h2.039v4.006z"/></svg>
    <button type="button" class="flash-close js-ajax-error-dismiss" aria-label="Dismiss error">
      <svg class="octicon octicon-x" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.48 8l3.75 3.75-1.48 1.48L6 9.48l-3.75 3.75-1.48-1.48L4.52 8 .77 4.25l1.48-1.48L6 6.52l3.75-3.75 1.48 1.48L7.48 8z"/></svg>
    </button>
    You can’t perform that action at this time.
  </div>


    
    <script crossorigin="anonymous" integrity="sha512-oJTLUPaOPb47Fbdk1MXyOw6oJiHT4bDh8oUf8xoNz2JjG4mgGE6xGWpaDj0Nfv0vE7X40m9MgTCgph7gWVR52w==" type="application/javascript" src="https://assets-cdn.github.com/assets/frameworks-a0a3b925d1c5d3f97657ed49211cbaec.js"></script>
    
    <script crossorigin="anonymous" async="async" integrity="sha512-tNd9rjeNpXuXVjx457bDgj7n//rIQCJSwh5EMCzZFWygRYfIvLq1iXILSsF7wHECAwy1yNhciDbcxLWNmM7H6Q==" type="application/javascript" src="https://assets-cdn.github.com/assets/github-7a46b4215fe0f5b9b5c0ca4a12070d88.js"></script>
    
    
    
  <div class="js-stale-session-flash stale-session-flash flash flash-warn flash-banner d-none">
    <svg class="octicon octicon-alert" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M8.893 1.5c-.183-.31-.52-.5-.887-.5s-.703.19-.886.5L.138 13.499a.98.98 0 0 0 0 1.001c.193.31.53.501.886.501h13.964c.367 0 .704-.19.877-.5a1.03 1.03 0 0 0 .01-1.002L8.893 1.5zm.133 11.497H6.987v-2.003h2.039v2.003zm0-3.004H6.987V5.987h2.039v4.006z"/></svg>
    <span class="signed-in-tab-flash">You signed in with another tab or window. <a href="">Reload</a> to refresh your session.</span>
    <span class="signed-out-tab-flash">You signed out in another tab or window. <a href="">Reload</a> to refresh your session.</span>
  </div>
  <div class="facebox" id="facebox" style="display:none;">
  <div class="facebox-popup">
    <div class="facebox-content" role="dialog" aria-labelledby="facebox-header" aria-describedby="facebox-description">
    </div>
    <button type="button" class="facebox-close js-facebox-close" aria-label="Close modal">
      <svg class="octicon octicon-x" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.48 8l3.75 3.75-1.48 1.48L6 9.48l-3.75 3.75-1.48-1.48L4.52 8 .77 4.25l1.48-1.48L6 6.52l3.75-3.75 1.48 1.48L7.48 8z"/></svg>
    </button>
  </div>
</div>

  <template id="site-details-dialog">
  <details class="details-reset details-overlay details-overlay-dark lh-default text-gray-dark" open>
    <summary aria-haspopup="dialog" aria-label="Close dialog"></summary>
    <details-dialog class="Box Box--overlay d-flex flex-column anim-fade-in fast">
      <button class="m-3 btn-octicon position-absolute right-0 top-0" type="button" aria-label="Close dialog" data-close-dialog>
        <svg class="octicon octicon-x" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.48 8l3.75 3.75-1.48 1.48L6 9.48l-3.75 3.75-1.48-1.48L4.52 8 .77 4.25l1.48-1.48L6 6.52l3.75-3.75 1.48 1.48L7.48 8z"/></svg>
      </button>
      <div class="octocat-spinner my-6 js-details-dialog-spinner"></div>
    </details-dialog>
  </details>
</template>

  <div class="Popover js-hovercard-content position-absolute" style="display: none; outline: none;" tabindex="0">
  <div class="Popover-message Popover-message--bottom-left Popover-message--large Box box-shadow-large" style="width:360px;">
  </div>
</div>

<div id="hovercard-aria-description" class="sr-only">
  Press h to open a hovercard with more details.
</div>


  </body>
</html>

