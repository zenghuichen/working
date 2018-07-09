using System;
using System.IO;
using System.Collections;
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
            string[] list = down.CreateNlist();
            //测试寻址性能
            string test = "n13044778";
            Console.WriteLine(test);
            int t = down.Find(test);
            Console.WriteLine(down.list[t]);
            Console.WriteLine(t.ToString());
            Console.ReadLine();
        }


    }

    public class ImageUpDown
    {
        public string[] list=new string[4000];
        public string GetOne(FileStream fs)
        {
            string result = "";

            return result;
        }
        public  string[] CreateNlist()
        {
            string[] list =new string[4000];
            //开始解析相关的程序代码
            FileStream fs = new FileStream(urllist.boxsyepath, FileMode.Open, FileAccess.Read);
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
                    i = i + 1;
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
                t = t + 1;
                index = t % 4000;
            }
            if (this.list[index] == null) return -1;
            if (this.list[index] == n) return index;
            return -1;
        }
    }
    public class meta
    {
        public string Synerl;
        public string id;
        public string url;
    }
    public class urllist
    {
        public static string boxurl = @"http://www.image-net.org/api/download/imagenet.bbox.synset?wnid=";
        public static string fall11path = @"E:\神经网络\imagenet_fall11_urls\fall11_urls.txt";
        public static string boxsyepath = @"E:\神经网络\imagenet_fall11_urls\boxlist.txt";
    }

}
