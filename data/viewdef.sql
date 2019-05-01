drop view if exists task;
create view task(taskID,taskName,tasktype,detail,postingdate,duration,technology,languages,prize,numsubmissions,numregistrants,diffdeg) as
select challengeID,challengeName,challengeType,detailedRequirements,
datediff(now(),postingDate),
duration,technology,languages,prize,numsubmissions,numregistrants,difficultyDegree from challenge_item
where currentStatus='Completed';# and numRegistrants>=10;

drop view if exists users;
create view users(handle,memberage,skills,competitionNum,submissionNum,winNum) as 
select handle, datediff(now(),memberSince),skills,competitionNums,submissionNums,winNums from user;

drop view if exists registration;
create view registration(taskid,handle,regdate) as 
select challengeID,handle,avg(datediff(now(),registrationDate)) from challenge_registrant group by challengeID,handle;

drop view if exists submission;
create view submission(taskid,handle,subnum,submitdate,score) as 
select challengeID,handle,count(*),avg(datediff(now(),submissionDate)),max(finalScore) from challenge_submission group by challengeID,handle;

select postingdate from task order by postingdate asc;
select * from users;
select * from task ;
select * from registration;
select * from submission;

