## Installing Apache Kafka on Windows

1. Install 7Zip. This is a really good unzip utility that  supports most compressed file types.

[Download 7Zip here](https://www.7-zip.org/download.html)

2. [Download kafka](https://kafka.apache.org/downloads)
Unzip this to a folder you will need to know where it is located.

I installed it to D:\java 

I then created the following batch file to run it

_start_kafka.bat
```
start D:\java\kafka_2.13-3.8.0\bin\windows\zookeeper-server-start.bat ..\config\zookeeper.properties
timeout 5
start D:\java\kafka_2.13-3.8.0\bin\windows\kafka-server-start.bat ..\config\server.properties
```

Notice 
1. I put a timeout after calling zookeeper just ot make sure it is started before kafka.
2. The zookeeper properties and server properties are required parameters. I just used the default files for this. You may need to customize these files.
