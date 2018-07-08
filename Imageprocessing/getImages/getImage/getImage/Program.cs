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
         
            FileStream fs = new FileStream(urllist.boxsyepath, FileMode.Open, FileAccess.Read);
            StreamReader sr = new StreamReader(fs);
            string r1 = "";
            while (true)
            {
                r1 = sr.ReadLine();
                if (r1 == null) break;
                int i = ImageUpDown.getxxxx(r1);
                Console.Write(i.ToString());
            }
            Console.WriteLine(r1);
            sr.Close();
            fs.Close();
            Console.ReadLine();
        }


    }

    public class ImageUpDown
    {
        public string getOne(FileStream fs)
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
                int i = getxxxx(r1);
            }
            sr.Close();
            fs.Close();
            return list;
        }
        public static int getxxxx(string xxxx)
        {
            int i = 0;
            string num = xxxx.Trim().Substring(1);
            i = int.Parse(num);
            return i;
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
