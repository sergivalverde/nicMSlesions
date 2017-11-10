@echo off
SET curdir=%CD%
SET cmd=ROBEX %~dp1%~nx1 %~dp2%~nx2 %~dp3%~nx3 %4
cd %~dp0
%cmd%
cd %curdir%
