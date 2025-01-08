+++
date = "2016-01-30T00:13:56Z"
title = "Project 2: File Explorer for android"

+++

In this project I am building a file explorer library for android.
As I was working on [catcher](https://github.com/ernan/catcher "Catcher") it became obvious I would need a file picker
and explorer solution. So I did a bit of looking on the web.
I found three interesting projects that nearly did what I wanted. I put a few of them together to come up with a hybrid solution.

<!--more-->

1. [FilePickerLibrary](https://github.com/DeveloperPaul123/FilePickerLibrary "FilePickerLibrary")  this is a great project looks beautiful 
lots of functionality great work.
2. [MaterialFilePicker](https://github.com/nbsp-team/MaterialFilePicker "MaterialFilePicker") this was really nice library, simple and the code was really clean. 
It is material so for now it has a modern android feel.
3. [FileBrowserView](https://github.com/psaravan/FileBrowserView "FileBrowserView")  this has a lot more functionality, 
it is a bit older but has all the stuff I want for this project.

So I spent the last couple of day merging these projects into a single hybrid. It is going pretty well but not finished yet.
You can find the current version of  this project here: [File Explorer](https://github.com/ernan/fileexplorer "File Explorer") 

 This will be an android library project. It is not the first time I have needed a file explorer for android so I am sure I will reuse it going forward.

## Some lessons learned during this

### 1. There is no wizard in Idea for a library project

I was a bit surprised here, I looked and tried a few things but really nothing worked.
In the end I just copied another library project template and renamed the paths.

Make sure and include the app and library in you base settings.gradle

```groovy

include ':app', ':library'

```



### 2. There is a simple way to include the library in your app for testing

 Intellij actually prompted me to do this which I thought was pretty cool. To add it just add it as a compile option

```groovy

dependencies {
    compile fileTree(include: ['*.jar'], dir: 'libs')
    testCompile 'junit:junit:4.12'
    compile 'com.android.support:appcompat-v7:23.1.1'
    compile 'com.android.support:design:23.1.1'
    compile project(':library')
}

```


### 3. You may have permissions issues initially during the project

  I found that when I transferred the apk to the emulator my permissions were not automatically applied. 
This was rather inconvenient. So i had to manually apply them in the emulator using the system tool

Here is some code (Android M4 +) to fix that:


```groovy

if (ActivityCompat.shouldShowRequestPermissionRationale(this, Manifest.permission.READ_EXTERNAL_STORAGE)) {
    DialogInterface.OnClickListener dialogClickListener = new DialogInterface.OnClickListener() {
        @Override
        public void onClick(DialogInterface dialog, int which) {
            switch (which) {
                case DialogInterface.BUTTON_POSITIVE:
                    ActivityCompat.requestPermissions(FileExplorerActivity.this,
                            new String[]{Manifest.permission.READ_EXTERNAL_STORAGE,
                                    Manifest.permission.WRITE_EXTERNAL_STORAGE},
                            REQUEST_FOR_READ_EXTERNAL_STORAGE);
                    break;
                case DialogInterface.BUTTON_NEGATIVE:
                    setResult(RESULT_CANCELED);
                    finish();
                    break;
            }
        }
    };
    AlertDialog.Builder builder = new AlertDialog.Builder(FileExplorerActivity.this);
    builder.setTitle(R.string.file_picker_permission_rationale_dialog_title)
            .setMessage(R.string.file_picker_permission_rationale_dialog_content)
            .setPositiveButton("Yes", dialogClickListener)
            .setNegativeButton("No", dialogClickListener).show();

}



```


### 4. My downloads directory on the emulator was empty so I had to transfer file

I used the adb to push a file to the emulator:

```bash

adb push NOTICE.txt /storage/emulated/0/Download/test.txt

```
