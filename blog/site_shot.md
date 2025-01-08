+++
date = "2016-02-03T20:46:38Z"
title = "Project 3: Site Shot"

+++


This is a really simple android application. It does one thing
It takes a photo of a web site and allows you to share the photo.

<!--more-->


You can find the code here [Site Shot](https://github.com/ernan/siteshot "Site Shot")

It is essentially android implementation of this blog post:

[https://shkspr.mobi/blog/2015/11/google-secret-screenshot-api/](https://shkspr.mobi/blog/2015/11/google-secret-screenshot-api/ "Google Screen Shot")

The cool thing about this app is it uses a lot of the posts I have already made.

This is the third project but is probably going to be the first release.

The app is very simple so I will do a quick overview of the useful functionality

The app uses [EventBus](http://github.com/greenrobot/EventBus "EventBus") to handle user events.

Once the user has entered url they can take a photo an dthis is where the magic happens 

The event handler function for a selected url calls the Google api

Code here is pretty straight forward 

1. Build the URL.
2. Process a get on a background thread, this is required in android
3. Call a handler function with the response data.   

```java

    public static void siteShot(final String request, final Response response, final Response errorResponse) {
        AsyncExecutor.create().execute(
                new AsyncExecutor.RunnableEx() {
                    @Override
                    public void run() throws Exception {
                        URL url = new URL(API + processRequest(request));
                        try {
                            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                            conn.setRequestMethod("GET");
                            conn.setRequestProperty("Accept", "application/json");
                            if (conn.getResponseCode() == 200) {
                                BufferedReader br = new BufferedReader(new InputStreamReader(
                                        conn.getInputStream()));
                                StringBuilder builder = new StringBuilder();
                                String output;
                                while ((output = br.readLine()) != null) {
                                    builder.append(output);
                                }
                                conn.disconnect();
                                response.handle(builder.toString());
                            } else {
                                errorResponse.handle("Failed : HTTP error code : "
                                        + conn.getResponseCode());
                            }
                        } catch (Exception e) {
                            errorResponse.handle(e.getMessage());
                        }
                    }
                }
        );
    }

```


Ok the next function turns that response data into an image file.

1. Select the data form the json response.
2. We have to do a little processing on this text before decoding it.
3. Base64 decode the data. 
4. Next we write this data to a file.
5. We then load this file into an image view and show a share button so the user can share save whatever the result image.

There is one more trick needed to share the image.

```java

  public void onEvent(String url) {
        Util.siteShot(url, new Response() {
                    @Override
                    public void handle(String response) {
                        try {
                            JSONObject obj = new JSONObject(response);
                            JSONObject screenshot = obj.getJSONObject("screenshot");
                            String data = new String(screenshot.get("data").toString());
                            data = data.replace("_", "/");
                            data = data.replace("-", "+");
                            byte[] decoded = Base64.decode(data, Base64.DEFAULT);
                            final String imagePath = getPictureFile();
                            BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(imagePath));
                            bos.write(decoded);
                            bos.flush();
                            bos.close();
                            runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    try {
                                        FileInputStream fis = new FileInputStream(new File(imagePath));
                                        iv.setImageBitmap(BitmapFactory.decodeStream(fis));
                                        fis.close();
                                        share.setVisibility(View.VISIBLE);
                                    } catch (Exception ex) {
                                        showError(ex.getMessage());
                                    }
                                }
                            });
                        } catch (Exception ex) {
                            showError(ex.getMessage());
                        }
                    }
                },
                new Response() {
                    @Override
                    public void handle(String errorMessage) {
                        showError(errorMessage);
                    }
                });
    }

```

Before we can share the file we need to write it to an external directory.
Seems weird but android will not allow external process have access to internal data in your application.

So this is the code to write the file to an external directory and the share form there is very simple.

```java

    public static String writeToExternal(Context context, String filename){
        String newFileName = null;
        try {
            File file = new File(context.getExternalFilesDir(null), filename);
            newFileName = file.getAbsolutePath();
            InputStream is = new FileInputStream(context.getFilesDir() + File.separator + filename);
            OutputStream os = new FileOutputStream(file);
            byte[] toWrite = new byte[is.available()];
            L.i("Available " + is.available());
            int result = is.read(toWrite);
            L.i("Result " + result);
            os.write(toWrite);
            is.close();
            os.close();
            L.i("Copying to " + context.getExternalFilesDir(null) + File.separator + filename);
            L.i("Copying from " + context.getFilesDir() + File.separator + filename);
        } catch (Exception e) {
            Toast.makeText(context, "File write failed: " + e.getLocalizedMessage(), Toast.LENGTH_LONG).show();
        }
        return newFileName;
    }


```