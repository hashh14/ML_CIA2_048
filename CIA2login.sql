use dbms;
create table login (name varchar(20), username varchar(20) unique, passcode varchar(20));
select * from login;
insert into login values ("Hashim", 