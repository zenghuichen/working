using System;
using System.IO;
using System.Collections;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace getImage
{
    class Program
    {
       
        static void Main(string[] args)
        {
            ImageUpDown down = new ImageUpDown();
            //开始测试读取和分类文件
            down.CreateNlist();//创建待选列表
            FileStream fs = new FileStream(Urllist.fall11path, FileMode.Open, FileAccess.Read);
            StreamReader sr = new StreamReader(fs);
            FileStream ifs = DirectoryImage.CreateIndex("main");
            StreamWriter isr = new StreamWriter(ifs);
            string temp = "";
            int index = -1;
            while (temp != null)
            {
                temp = sr.ReadLine();
                Meta meta = ImageUpDown.GetMetas(temp);
                //判定这个对象
                index = down.Find(meta.Synerl);
                if (index == -1) continue;
                if (down[index] == meta.Synerl)//表示找到需要将其提取出来
                {
                    isr.WriteLine(meta.Print());
                }
            }
            isr.Close();
            ifs.Close();
            sr.Close();
            fs.Close();
            Console.ReadLine();
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
            return meta;
        }
        string[] list=new string[4000];
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
            string impath = Urllist.datapath + "\\" + Filename + "\\" + "image";//image
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
    }

}
