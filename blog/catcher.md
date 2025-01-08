+++
date = "2016-01-25T22:06:44Z"
title = "Project 1: Catcher"

+++


This is an android application to transfer files from your phone to somewhere else. I will be built as a PC solution but
can be used for a server solution also. 
<!--more-->


So I was trying to transfer some documents to my android phone and could not find an easy way to do this. 
In the end I transferred to my [onedrive](https;//onedrive.com) and then opened up one the android [Microsoft Onedrive](https://play.google.com/store/apps/details?id=com.microsoft.skydrive&hl=en) 
application and downloaded the file to my android device.

There are a lot of ways to get data on to and off your android device. 
For this first project I am going to build one more. 
The reason I am building this is I am going to make a lot of use of it in some upcoming projects.

You can find the current version of this project here: [catcher](https://github.com/ernan/catcher "Catcher")


This will be an android application. I have been programming Android for a while now. 
So I have  a few libraries I like to reuse. 
I will go over them in this blog post


1. compile 'de.greenrobot:eventbus:2.4.0'
2. compile 'de.greenrobot:greendao:1.3.7'

These two libraries simplify android development a lot. In every android project I just automatically include them.

[EventBus](http://github.com/greenrobot/EventBus "EventBus")

is a very simple to use event bus for android. I have been programming android for a long time now and 
honestly I am still not sure what is the best way to communicate between activities, fragments, services, controls etc.
Well with EventBus I don't have to think about this stuff, just use the bus. 
It works it is really simple has every option you could need and is super fast.

[greenDAO](http://github.com/greenrobot/greenDAO "greenDAO")
is an ORM. This simplifies database development on android.
There are a few quirks to using it which I will go over in upcoming posts. Anyway key point this is an android application I don't want to 
spend any time thinking about the data layer I want to get on to building the application. 

There are a couple of other libraries I am including in this application

[jeromq](https://github.com/zeromq/jeromq "jeromq")

This whole project is going to be built on this so I will be coming back to this later. 
 
[Bolts](https://github.com/BoltsFramework/Bolts-Android "Bolts")
This is new. I want to use this to handle the threading for this application.
Before this I used [RXAndroid](https://github.com/ReactiveX/RxAndroid "RxAndroid") 
why the change? I found it a bit awkward and clumsy. 
Well lets see I could end up eating my words before very long.


Anyway the build.gradle is as follows

```groovy
apply plugin: 'com.android.application'
apply plugin: 'android-apt'

android {
    compileSdkVersion 23
    buildToolsVersion "23.0.2"

    defaultConfig {
        applicationId "ie.programmer.catcher"
        minSdkVersion 19
        targetSdkVersion 23
        versionCode 1
        versionName "1.0"
        multiDexEnabled true
    }
    buildTypes {
        release {
            minifyEnabled false
            proguardFiles getDefaultProguardFile('proguard-android.txt'), 'proguard-rules.pro'
        }
    }
}

dependencies {
    compile fileTree(dir: 'libs', include: ['*.jar'])
    testCompile 'junit:junit:4.12'
    compile 'com.android.support:appcompat-v7:23.1.0'
    compile 'com.android.support:design:23.1.0'

    compile 'org.zeromq:jeromq:0.3.5'
    compile 'de.greenrobot:eventbus:2.4.0'
    compile 'de.greenrobot:greendao:1.3.7'
    compile 'commons-io:commons-io:2.4'
    compile 'com.path:android-priority-jobqueue:1.1.2'
    compile 'com.parse.bolts:bolts-tasks:1.3.0'
    compile 'com.parse.bolts:bolts-applinks:1.3.0'
    compile 'org.androidannotations:androidannotations-api:3.3.2'
    compile('com.github.afollestad.material-dialogs:core:0.8.5.0@aar') {
        transitive = true
    }
    apt 'org.androidannotations:androidannotations:3.3.2'
}

```