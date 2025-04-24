# Astro_Collaboration
This is a repository to be used in the collaborative effort of developing an Einstein Toolkit module for the FERN code. 

Basics notes on E-T:


Let's start with a brief overview of what a thorn needs to include:

a Cactus thorn must have a name.
a Cactus thorn must live in an arrangement.
a Cactus thorn must have four ccl (Cactus Configuration Language) files:
interface.ccl
schedule.ccl
param.ccl
configuration.ccl
a Cactus thorn must have a src directory
a Cactus thorn must have a make.code.defn file in that source directory
The above thorn files and directories should reside in Cactus/arrangements/ArrangementName/ThornName/.
