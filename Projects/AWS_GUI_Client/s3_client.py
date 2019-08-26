#! /usr/bin/python3
from tkinter import *
from tkinter.filedialog import askdirectory
from os import listdir, remove, execl
from shutil import rmtree, make_archive
from getpass import getuser, getpass
from os.path import isdir, basename
from time import sleep
from sys import executable, argv
try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError as e:
    print("Unable to import boto3\n%s" % e)
    exit()


class S3Zilla:
    def __init__(self, master):
        # self.service_name = 's3'
        # self.use_ssl = False
        # self.endpoint_url = 'http://10.5.41.189:9090'
        # self.ak = 'O3X91G71GPGA4QG36V28'
        # self.sk = 'qA1FpiOuNUqDxq3vOiTZDispICubnLi29PRuexqG'
        
        ### 测试
        self.service_name = 's3'
        self.use_ssl = False
        self.endpoint_url = 'http://10.5.41.14:7480'
        self.ak = 'HH4QU2FLODUU5P991G47'
        self.sk = 'WrnVmVf9CpAwrWR5CPWELmAvksjeyYMqn1koY4q0'
        error_msg = "Ensure S3 is configured on your machine:"
        try:
            self.s3 = boto3.resource(service_name=self.service_name,
                                     use_ssl=self.use_ssl,
                                     endpoint_url=self.endpoint_url,   
                                     aws_access_key_id = self.ak,
                                     aws_secret_access_key = self.sk)
        except Exception as e:
            print("%s: %s" % (error_msg, e))
            exit(1)
        try:
            self.s3c = boto3.client(service_name=self.service_name,
                                     use_ssl=self.use_ssl,
                                     endpoint_url=self.endpoint_url,   
                                     aws_access_key_id = self.ak,
                                     aws_secret_access_key = self.sk)
        except Exception as err:
            print("%s: %s" % (error_msg, err))
            exit(1)
        self.colors = {
            'light-grey': '#D9D9D9',
            'blue': '#2B547E',
            'black': '#000000',
            'red': '#FF3346',
            'grey': '#262626',
            'cyan': '#80DFFF'
        }
        self.master = master
        self.master.title("商汤科技 aws S3 File Transfer Client")
        self.master.configure(bg=self.colors['grey'])
        self.master.geometry("885x645")
        menu = Menu(self.master)
        menu.config(
            background=self.colors['grey'],
            fg=self.colors['light-grey']
        )
        self.master.config(menu=menu)
        file = Menu(menu)
        file.add_command(
            label="Exit",
            command=self.quit
        )
        menu.add_cascade(
            label="File",
            menu=file
        )
        refresh = Menu(menu)
        refresh.add_command(
            label="Local",
            command=self.refresh_local
        )
        refresh.add_command(
            label="S3",
            command=self.refresh_s3
        )
        menu.add_cascade(label="Refresh", menu=refresh)
        self.dir, self.drp_sel, self.bucket_name = '', '', ''
        self.folder_path = StringVar()
        self.dropdown = StringVar()
        self.dropdown_data = self.populate_dropdown()
        if not self.dropdown_data:
            self.dropdown_data = ['none available']
        self.deleted = False
        self.local_sel, self.s3_sel = ([] for i in range(2))
        self.title_label = Label(
            master,
            fg=self.colors['light-grey'],
            bg=self.colors['grey'],
            font="Helvetica 10 bold",
            width=120
        )
        self.local_label = Label(
            master,
            fg=self.colors['light-grey'],
            bg=self.colors['grey'],
            text="LOCAL FILE SYSTEM",
            font="Helvetica 10 bold underline",
            width=60
        )
        self.s3_label = Label(
            master,
            fg=self.colors['light-grey'],
            bg=self.colors['grey'],
            text="AMAZON  S3",
            font="Helvetica 10 bold underline",
            underline=True,
            width=60
        )
        self.dropdown_box = OptionMenu(
            master,
            self.dropdown,
            *self.dropdown_data,
            command=self.set_drop_val
        )
        self.dropdown_box.configure(
            fg=self.colors['light-grey'],
            bg=self.colors['blue'],
            width=27,
            highlightbackground=self.colors['black'],
            highlightthickness=2
        )
        self.browse_button = Button(
            master,
            fg=self.colors['light-grey'],
            bg=self.colors['blue'],
            text="Browse",
            width=30,
            highlightbackground=self.colors['black'],
            highlightthickness=2,
            command=self.load_dir
        )
        self.browse_label = Label(
            master,
            fg=self.colors['light-grey'],
            bg=self.colors['grey'],
            text="No directory selected",
            width=37,
            font="Helvetica 10"
        )
        self.bucket_label = Label(
            master,
            fg=self.colors['light-grey'],
            bg=self.colors['grey'],
            text="No bucket selected",
            width=37,
            font="Helvetica 10"
        )
        self.refresh_btn_local = Button(
            master,
            fg=self.colors['light-grey'],
            bg=self.colors['blue'],
            text="REFRESH",
            width=30,
            highlightbackground=self.colors['black'],
            highlightthickness=2,
            command=self.refresh_local
        )
        self.refresh_btn_s3 = Button(
            master,
            fg=self.colors['light-grey'],
            bg=self.colors['blue'],
            text="REFRESH",
            width=30,
            highlightbackground=self.colors['black'],
            highlightthickness=2,
            command=self.refresh_s3
        )
        self.explorer_label_local = Label(
            master,
            fg=self.colors['light-grey'],
            bg=self.colors['blue'],
            width=30,
            text="Local File System:  "
        )
        self.explorer_label_s3 = Label(
            master,
            fg=self.colors['light-grey'],
            bg=self.colors['black'],
            width=30,
            text="S3 File System"
        )
        self.ex_loc = Listbox(
            master,
            fg=self.colors['cyan'],
            bg=self.colors['black'],
            width=49,
            height=18,
            highlightcolor=self.colors['black'],
            selectmode="multiple"
        )
        self.ex_s3 = Listbox(
            master,
            fg=self.colors['cyan'],
            bg=self.colors['black'],
            width=49,
            height=18,
            highlightcolor=self.colors['black'],
            selectmode="multiple"
        )
        self.upload_button = Button(
            master,
            fg=self.colors['light-grey'],
            bg=self.colors['blue'],
            text="Upload ->",
            width=20,
            highlightbackground=self.colors['black'],
            highlightthickness=2,
            command=self.upload
        )
        self.download_button = Button(
            master,
            fg=self.colors['light-grey'],
            bg=self.colors['blue'],
            text="<- Download",
            width=20,
            highlightbackground=self.colors['black'],
            highlightthickness=2,
            command=self.download
        )
        self.delete_local = Button(
            master,
            fg=self.colors['light-grey'],
            bg=self.colors['red'],
            text="DELETE",
            width=20,
            highlightbackground=self.colors['black'],
            command=self.delete_local_records
        )
        self.delete_s3 = Button(
            master,
            fg=self.colors['light-grey'],
            bg=self.colors['red'],
            text="DELETE",
            width=20,
            highlightbackground=self.colors['black'],
            command=self.delete_s3_records
        )
        self.found_label_local = Label(
            master,
            fg=self.colors['light-grey'],
            bg=self.colors['grey'],
            text="found local",
            width=54
        )
        self.found_label_s3 = Label(
            master,
            fg=self.colors['light-grey'],
            bg=self.colors['grey'],
            text="found s3",
            width=54
        )
        self.status_label = Label(
            master,
            fg=self.colors['light-grey'],
            bg=self.colors['grey'],
            text="Hello " + getuser(),
            width=54
        )
        self.create_bucket_label = Label(
            master,
            fg=self.colors['light-grey'],
            bg=self.colors['grey'],
            text="New Bucket:",
        )
        self.create_bucket_name = Text(
            master,
            fg=self.colors['cyan'],
            bg=self.colors['black'],
            width=25,
            height=1
        )
        self.create_bucket_button = Button(
            master,
            fg=self.colors['light-grey'],
            bg=self.colors['blue'],
            text="Create",
            width=5,
            highlightbackground=self.colors['black'],
            highlightthickness=2,
            command=self.create_bucket
        )
        ## Chiyuan Custom
        self.create_dir_label = Label(
            master,
            fg=self.colors['light-grey'],
            bg=self.colors['grey'],
            text="New Folder:",
        )
        self.create_dir_name = Text(
            master,
            fg=self.colors['cyan'],
            bg=self.colors['black'],
            width=15,
            height=1
        )
        self.create_dir_button = Button(
            master,
            fg=self.colors['light-grey'],
            bg=self.colors['blue'],
            text='Create',
            width=3,
            highlightbackground=self.colors['black'],
            highlightthickness=2,
            command=self.create_dir
        )

        # ####### begin grid placement ####### #
        self.title_label.grid(
            row=0,
            sticky=E+W,
            padx=20,
            pady=5
        )
        self.local_label.grid(
            row=0,
            sticky=W,
            padx=8,
            pady=5
        )
        self.s3_label.grid(
            row=0,
            sticky=E,
            padx=0,
            pady=5
        )
        self.browse_button.grid(
            row=1,
            sticky=W,
            padx=86,
            pady=10
        )
        self.dropdown_box.grid(
            row=1,
            sticky=E,
            padx=86,
            pady=5
        )
        self.browse_label.grid(
            row=2,
            sticky=W,
            padx=86,
            pady=5
        )
        self.bucket_label.grid(
            row=2,
            sticky=E,
            padx=86,
            pady=5
        )
        self.refresh_btn_local.grid(
            row=3,
            sticky=W,
            padx=86,
            pady=10
        )
        self.refresh_btn_s3.grid(
            row=3,
            sticky=E,
            padx=86,
            pady=10
        )
        self.explorer_label_local.grid(
            row=4,
            sticky=W,
            padx=20
        )
        self.explorer_label_s3.grid(
            row=4,
            sticky=E,
            padx=20
        )
        self.ex_loc.grid(
            row=4,
            sticky=W,
            padx=20
        )
        self.ex_s3.grid(
            row=4,
            sticky=E,
            padx=20
        )
        self.upload_button.grid(
            row=5,
            sticky=W,
            padx=224,
            pady=0
        )
        self.download_button.grid(
            row=5,
            sticky=E,
            padx=224,
            pady=0
        )
        self.delete_local.grid(
            row=5,
            sticky=W,
            padx=20,
            pady=0
        )
        self.delete_s3.grid(
            row=5,
            sticky=E,
            padx=20,
            pady=0
        )
        self.found_label_local.grid(
            row=6,
            sticky=W,
            padx=0,
            pady=20
        )
        self.found_label_s3.grid(
            row=6,
            sticky=E,
            padx=0,
            pady=20
        )
        self.status_label.grid(
            row=7,
            sticky=W,
            padx=0,
            pady=20
        )
        self.create_bucket_label.grid(
            row=7,
            sticky=E,
            padx=330,
            pady=0
        )
        self.create_bucket_name.grid(
            row=7,
            sticky=E,
            padx=100,
            pady=0
        )
        self.create_bucket_button.grid(
            row=7,
            sticky=E,
            padx=20,
            pady=0
        )
        ## Chiyuan Customize
        self.create_dir_label.grid(
            row=8,
            sticky=E,
            padx=330,
            pady=0
        )
        self.create_dir_name.grid(
            row=8,
            sticky=E,
            padx=100,
            pady=0
        )
        self.create_dir_button.grid(
            row=8,
            sticky=E,
            padx=20,
            pady=0
        )
        n1 = "%s files found" % str(self.ex_loc.size())
        self.set_found_local_label(n1)
        n2 = "%s files found" % str(self.ex_s3.size())
        self.set_found_s3_label(n2)

    def quit(self):
        exit()

    def get_local_sel(self):
        return [self.ex_loc.get(i) for i in self.ex_loc.curselection()]

    def get_s3_sel(self):
        return [self.ex_s3.get(i) for i in self.ex_s3.curselection()]

    def set_drop_val(self, selection):
        self.drp_sel = selection

    def delete_local_records(self):
        files = self.get_local_sel()
        if not files:
            message = "Please select a file(s) to delete"
            self.set_status_label(message)
        else:
            self.del_local(files)

    def del_local(self, files_remaining):
        if len(files_remaining) > 0:
            f = files_remaining.pop(0)
            if not isdir(self.dir + "/" + f):
                try:
                    remove("%s/%s" % (self.dir, f))
                except Exception as err:
                    self.set_status_label("%s" % err)
                    self.status_label.update_idletasks()
                self.del_local(files_remaining)
            else:
                try:
                    rmtree("%s/%s" % (self.dir, f))
                except Exception as e:
                    self.set_status_label("%s" % e)
                    self.status_label.update_idletasks()
                self.del_local(files_remaining)
        self.deleted = True
        self.refresh_local()

    def delete_s3_records(self):
        removal = ''
        if not self.drp_sel:
            m = "Please select a bucket..."
            self.set_status_label(m)
        else:
            removal = self.get_s3_sel()
        if not removal:
            m = "Please select at least 1 object to delete"
            self.set_status_label(m)
        else:
            bucket = self.s3.Bucket(self.drp_sel)
            for rm in removal:
                for k in bucket.objects.all():
                    if k.key != rm:
                        continue
                    k.delete()
                    break
            self.deleted = True
            self.refresh_s3()

    def load_dir(self):
        self.dir = askdirectory()
        self.set_local_browse_label(self.dir)
        self.refresh_local()

    def refresh_local(self):
        if not self.dir:
            m = "Use the browse button to select a directory"
            self.set_status_label(m)
        else:
            self.set_local_browse_label(self.dir)
            self.ex_loc.delete(0, 'end')
            x = self.dir + "/"
            d = [f if not isdir(x+f) else f + '/' for f in sorted(listdir(x))]
            self.ex_loc.insert('end', *d)
            if not self.deleted:
                m = "Hello %s" % getuser()
            else:
                m = "FINISHED DELETING"
                self.deleted = False
            self.set_status_label(m)
            n = "%s files found" % str(self.ex_loc.size())
            self.set_found_local_label(n)

    def refresh_s3(self):
        if 'none available' in self.dropdown_data:
            m = "Please create at least one S3 bucket"
            self.set_status_label(m)
        elif not self.drp_sel:
            m = "Please select a bucket from the drop-down list"
            self.set_status_label(m)
        else:
            self.ex_s3.delete(0, 'end')
            self.ex_s3.insert('end', *self.get_bucket_contents())
            self.set_status_label("Hello %s" % getuser())
            self.set_s3_bucket_label(self.drp_sel)
            n = "%s files found" % str(self.ex_s3.size())
            self.set_found_s3_label(n)
            self.found_label_s3.update_idletasks()
            if not self.deleted:
                m = "Hello %s" % getuser()
            else:
                m = "FINISHED DELETING"
                self.deleted = False
            self.set_status_label(m)

    def finished(self, incoming_message):
        d = "FINISHED %s" % incoming_message
        for letter in enumerate(d):
            self.set_status_label(d[0:letter[0] + 1])
            self.status_label.update_idletasks()
            sleep(.1)

    def upload(self):
        if not self.drp_sel or not self.dir:
            m = "Ensure a local path and S3 bucket are selected"
            self.set_status_label(m)
        elif not self.get_local_sel():
            m = "Ensure files are selected to upload"
            self.set_status_label(m)
        else:
            for selection in self.get_local_sel():
                file_ = "%s/%s" % (self.dir, selection)
                if not isdir(file_):
                    self.s3c.upload_file(file_, self.drp_sel, basename(file_))
                else:
                    zipd = make_archive(file_, 'zip', self.dir, selection)
                    self.s3c.upload_file(zipd, self.drp_sel, basename(zipd))
                    remove(zipd)
                m = "Uploaded: %s" % selection
                self.set_status_label(m)
                self.status_label.update_idletasks()
            self.refresh_s3()
            self.finished("UPLOAD")

    def download(self):
        if not self.drp_sel or not self.dir:
            m = "Ensure a file and bucket have been selected"
            self.set_status_label(m)
        elif not self.get_s3_sel():
            m = "Ensure files are selected to download"
            self.set_status_label(m)
        else:
            for selection in self.get_s3_sel():
                file_ = "%s/%s" % (self.dir, selection)
                self.s3c.download_file(self.drp_sel, selection, file_)
            self.refresh_local()
            self.finished("DOWNLOAD")

    def get_bucket_contents(self):
        bucket = self.s3.Bucket(self.drp_sel)
        return [s3_file.key for s3_file in bucket.objects.all()]

    def populate_dropdown(self):
        return [bucket.name for bucket in self.s3.buckets.all()]

    def set_local_browse_label(self, incoming):
        if len(incoming) > 35:
            self.browse_label.config(text=basename(incoming) + '/')
        else:
            self.browse_label.config(text=incoming)

    def set_s3_bucket_label(self, incoming):
        self.bucket_label.config(text=incoming)

    def set_status_label(self, incoming):
        self.status_label.config(text=incoming)

    def set_found_local_label(self, incoming):
        self.found_label_local.config(text=incoming)

    def set_found_s3_label(self, incoming):
        self.found_label_s3.config(text=incoming)

    def create_bucket(self):
        self.bucket_name = self.create_bucket_name.get("1.0", END).strip()
        if not self.bucket_name:
            m = "Please enter a new bucket name"
            self.set_status_label(m)
        else:
            pre_exists = False
            try:
                self.s3.create_bucket(Bucket=self.bucket_name)
            except ClientError as ce:
                pre_exists = True
                m = "Bucket name is already in use. "
                m += "Choose a different name."
                self.set_status_label(m)
            if not pre_exists:
                m = "%s created: restarting..." % self.bucket_name
                self.set_status_label(m)
                self.status_label.update_idletasks()
                res = executable
                execl(res, res, *argv)

    # Chiyuan Customize
    def create_dir(self):
        new_path = self.create_dir_name.get("1.0", END).strip()
        if not new_path:
            m = "Please enter a new path"
            self.set_status_label(m)
        else:
            bucket = self.s3.Bucket(self.drp_sel)
            bucket.put_object(Bucket=bucket.name, Key=new_path)
            self.refresh_s3()


if __name__ == "__main__":
    root = Tk()
    s3_zilla = S3Zilla(root)
    root.mainloop()
