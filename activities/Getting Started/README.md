# Prerequisites and Setting things up

## Login to your Virtual Machine 

1. Go to the following URL https://training.datacouch.io/pluralsight/#/
2. Login using the information given to you in class

![image-20241002104143946](images/image-20241002104143946.png)

3. You should see a similar screen once you are logged in

![image-20241002105141115](images/image-20241002105141115.png)

## Download the GitHub Repo and Start Jupyter Lab

1. Double click on the `MATE Terminal` icon to launch a terminal

<img src="images/image-20241002105242342.png" alt="image-20241002105242342" style="zoom:50%;" />

3. Navigate to Desktop 

```bash
cd Desktop/
```

4. From Desktop folder type the following to download the GitHub repo for this class

```bash
git clone https://github.com/tatwan/local_llms.git
```

![image-20241002105545314](images/image-20241002105545314.png)

5. Go to the folder

```bash
cd local_llms/
```

6. Start Jupyter Lab

```bash
jupyter lab
```

![image-20241002105703392](images/image-20241002105703392.png)

This should launch FireFox to http://localhost:8888/lab

![image-20241002105826125](images/image-20241002105826125.png)



## Download Llamfile

1. Go to the following URL using Firefox  from inside the Virtual Machine https://github.com/Mozilla-Ocho/llamafile
2. Scroll down to the **Other example llamafiles** section 

<img src="images/image-20241002104503351.png" alt="image-20241002104503351" style="zoom:50%;" />

3. Click on the `TinyLlama-1.1B-Chat-v1.0.F16.llamafile` file to download inside the VM. You can select the default **Downloads** folder
4. Open a new Terminal
5. Navigate to the Downloads folder `cd Downloads/`
6. Make sure you see the file. Type `ls` and you should see the TinyLlama file similar to the following 

![image-20241002110142795](images/image-20241002110142795.png)

7. Type the following to gran permissions 

```bash
chmod +x TinyLlama-1.1B-Chat-v1.0.F16.llamafile
```

8. To start the llamafile server type 

```bash
./TinyLlama-1.1B-Chat-v1.0.F16.llamafile
```

9. In FireFox it should launch the following page on http://127.0.0.1:8080 

<img src="images/image-20241002110318840.png" alt="image-20241002110318840" style="zoom:80%;" />







