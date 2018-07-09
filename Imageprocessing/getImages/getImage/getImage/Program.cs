using System;
using System.IO;
using System.Collections;
using System.Runtime.InteropServices;
using System.Threading;
using System.Drawing;
using System.Drawing.Imaging;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Net;
namespace getImage
{
    class Program
    {

        static void Main(string[] args)
        {
            MuiltThreadDown muiltThreadDown = new MuiltThreadDown();
            muiltThreadDown.main_d();
            Console.ReadLine();
        }


    }

    public class MuiltThreadDown
    {
        Thread read_th;//读取文件的线程
        Thread log_th;
        MOList<Meta> list = new MOList<Meta>();
        MOList<string> loglist = new MOList<string>();//等待输出到log队列
        bool reading = true;
        bool downing = true;
        bool logging = true;
        public void Readtxt()
        {
            FileStream fs = new FileStream(Urllist.indexpath + "\\main", FileMode.OpenOrCreate, FileAccess.ReadWrite);
            StreamReader sr = new StreamReader(fs);
            string temp = "";
            while (temp != null)
            {
                reading = true;
                temp = sr.ReadLine();
                if (temp == null) break;
                string[] vs = temp.Split(',');
                Meta meta = new Meta(vs[0], vs[1], vs[2], vs[3]);
                while (list.Push(meta) <= 0){ }

            }
            sr.Close();
            fs.Close();
            reading = false;
        }
        public void Loadtxt()
        {
            while (!list.isEmpty())
            {
                downing = true;
                Meta meta = null;
                lock (this)
                {
                    meta = list.Pop();
                }
                //在线下载图片并输出到指定的文件夹中的
                downFile(meta);

            }
            downing = false;
        }
        public int downFile(Meta meta)
        {
            try
            {
                WebRequest request = WebRequest.Create(meta.url);
                WebResponse response = request.GetResponse();
                Stream reader = response.GetResponseStream();
                string path = Urllist.datapath + "\\"+meta.path;
                FileStream writer = new FileStream(path, FileMode.OpenOrCreate, FileAccess.Write);
                byte[] buff = new byte[1024*5];
                int c = 0; //实际读取的字节数
                bool hasdata = false;
                while ((c = reader.Read(buff, 0, buff.Length)) > 0)
                {
                    writer.Write(buff, 0, c);
                    hasdata = true;
                }
                writer.Close();
                if (!hasdata)
                {
                    loglist.Push(meta.Print()+","+"没有收到数据");
                }
                return 1;
            }
            catch(Exception ex)
            {
                loglist.Push(meta.Print()+","+ex.Message.ToString());
            }
            return -1;
        }
        public void Logtext()
        {
            string path = Urllist.mainurl + "\\log.txt";
            StreamWriter sw = null;
            if (!File.Exists(path))
            {
                FileStream fs = new FileStream(path, FileMode.OpenOrCreate, FileAccess.ReadWrite);
                fs.Close();
            }
            sw = new StreamWriter(path, true);
            while (!loglist.isEmpty())
            {
                logging = true;
                string d = loglist.Pop();
                sw.WriteLine(d);
            }
            sw.Close();
            logging = false;
        }
        public void main_d()
        {
            //参数初始化
            ImageUpDown down = new ImageUpDown();
            read_th = new Thread(Readtxt);
            log_th = new Thread(Logtext);
            Thread[] downs = new Thread[10];
            read_th.Start();
            log_th.Start();
            //开始下载文件
            while (reading || downing || logging) 
            {
                for (int i = 0; i < downs.Length; i++)
                {
                        if (downs[i]==null || downs[i].ThreadState == ThreadState.Stopped)
                        {
                            downs[i] = new Thread(Loadtxt);
                            downs[i].Start();
                        }
                }

                if (!logging)
                {
                    if (log_th.ThreadState == ThreadState.Stopped)
                    {
                        log_th = new Thread(Logtext);
                        log_th.Start();
                    }
                }
            }
            //清理一波小尾巴
            Console.WriteLine("线程挂起");
            Thread.Sleep(1000);
            Console.WriteLine("线程重载");
            if (!list.isEmpty())
            {
                for (int i = 0; i < downs.Length; i++)
                {
                    if (downs[i] == null || downs[i].ThreadState == ThreadState.Stopped)
                    {
                        downs[i] = new Thread(Loadtxt);
                        downs[i].Start();
                    }
                }
            }
            if (!loglist.isEmpty())
            {
                if (log_th.ThreadState == ThreadState.Stopped)
                {
                    log_th = new Thread(Logtext);
                    log_th.Start();
                }
            }
            Console.WriteLine("\n程序完成");
        }
    }
    public class ImageUpDown
    {
        public const int step =1;
        public static Meta GetMetas(string rows)
        {
            Meta meta = new Meta();
            //得到其中的某些单独文件
            string[] vs = rows.Split('\t');
            //解析其中的对象
            string[] nName = vs[0].Split('_');
            meta.Synerl = nName[0];
            meta.id = nName[1];
            meta.url = vs[1];
            meta.path = meta.Synerl + "\\images" + meta.id + ".jpg";
            return meta;
        }
        string[] list=new string[4000];
        public int Length { get { return 4000; } }
        public string GetOne(FileStream fs)
        {
            string result = "";

            return result;
        }
        public  string[] CreateNlist()
        {
            string[] list =new string[4000];
            //开始解析相关的程序代码
            FileStream fs = new FileStream(Urllist.boxsyepath, FileMode.Open, FileAccess.Read);
            StreamReader sr = new StreamReader(fs);
            string r1 = "";
            while (true)
            {

                r1 = sr.ReadLine();
                if (r1 == null) break;
                int i = Getxxxx(r1);//
                int index = i % 4000;
                while (list[index]!=null)
                {
                    i = i + step;
                    index = i % 4000;
                }
                list[index] = r1;
            }
            sr.Close();
            fs.Close();
            this.list = list;
            return list;
        }
        public static int Getxxxx(string xxxx)
        {
            int i = 0;
            string num = xxxx.Trim().Substring(1);
            i = int.Parse(num);
            return i;
        }
        public int Find(string n)
        {
            int t = Getxxxx(n);
            int index = t%4000;
            while (n != this.list[index]&&this.list[index]!=null)
            {
                t = t + step;
                index = t % 4000;
            }
            if (this.list[index] == null) return -1;
            if (this.list[index] == n) return index;
            return -1;
        }
        public string this[int i]
        {
            get {
                return list[i];
            }
        }
    }
    public class DirectoryImage
    {
        public static ArrayList CreateDataDir(string Filename)
        {
            ArrayList result = new ArrayList();
            string impath = Urllist.datapath + "\\" + Filename + "\\" + "images";//image
            string annnotpath = Urllist.datapath + "\\" + Filename + "\\" + "annotion";//box
            DirectoryInfo info1 = System.IO.Directory.CreateDirectory(impath); result.Add(info1);
            DirectoryInfo info2 = System.IO.Directory.CreateDirectory(annnotpath); result.Add(info2);
            return result;
        }
        public static FileStream  CreateIndex(string Filename)
        {
            string fPath = Urllist.indexpath+"\\"+Filename;
            return new FileStream(fPath, FileMode.OpenOrCreate, FileAccess.ReadWrite);
        }
    }
    public class Meta
    {
        public string Synerl;
        public string id;
        public string url;
        public string path;
        public Meta()
        {
            this.Synerl = "";
            this.id = "";
            this.url = "";
            this.path = "*";
        }
        public Meta(string s, string i, string u, string p)
        {
            this.Synerl = s;
            this.id = i;
            this.url = u;
            this.path = p;
        }
        public string Print()
        {
            return this.Synerl + "," + this.id + "," + this.url + "," + this.path;
        }
    }
    [StructLayout(LayoutKind.Sequential)]
    public struct StoreMeta//存储单元-->固定长度
    {
        
    }
    public class Urllist
    {
        public static string boxurl = @"http://www.image-net.org/api/download/imagenet.bbox.synset?wnid=";
        public static string fall11path = @"E:\data\imagenet\imagenet_fall11_urls\fall11_urls.txt";
        public static string boxsyepath = @"E:\data\imagenet\imagenet_fall11_urls\boxlist.txt";
        public static string datapath = @"E:\data\imagenet\imagenet_fall11_urls\data";//数据文件---》图片
        public static string indexpath = @"E:\data\imagenet\imagenet_fall11_urls\index";//索引文件夹
        public static string mainurl = @"E:\data\imagenet\imagenet_fall11_urls";
    }
    public class MOList<T>
    {
        T[] t;
        int L = 0;
        int H = 0;
        public int Length;
        public MOList()
        {
            t = new T[4000];
        }
        public int Push(T temp)
        {
            if (!isPush()) return -1;//表示不能添加
            t[H++ %4000] = temp;
            return H;
        }
        public T Pop()
        {
            Length--;
            return t[L++ % 4000];
        }
        public bool isPush()
        {
            Length++;
            if ((L % 4000) == (H + 1) % 4000)
            {
                return false;//表示队列已满
            }
            else
            {
                return true;//队列为空
            }
        }
        public bool isEmpty()
        {
            if (L % 4000 == H % 4000)
            {
                return true;
            }
            else
            {
              return   false;
            }
        }
    }

}
