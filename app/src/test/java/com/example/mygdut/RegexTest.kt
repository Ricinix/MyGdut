package com.example.mygdut

import kotlinx.coroutines.runBlocking
import org.junit.Test
import java.util.*

class RegexTest {
    @Test
    fun classTable_regex_test(){
        val content = "\n" +
                "\n" +
                "\n" +
                "\n" +
                "\n" +
                "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\" \"http://www.w3.org/TR/html4/loose.dtd\">\n" +
                "<html>\n" +
                "<head>\n" +
                "<meta http-equiv=\"pragma\" content=\"no-cache\" /> \n" +
                "<meta http-equiv=\"cache-control\" content=\"no-cache\" /> \n" +
                "<meta http-equiv=\"expires\" content=\"0\" /> \n" +
                "<meta http-equiv=\"Content-Type\" content=\"text/html; charset=UTF-8\" /> \n" +
                "<title>学生个人学期课表</title>\n" +
                "<meta http-equiv=\"content-type\" content=\"text/html; charset=UTF-8\" />\n" +
                "<meta name=\"renderer\" content=\"webkit\">\n" +
                "<meta name=\"ctxPath\" content=\"/\">\n" +
                "<link rel=\"shortcut icon\" href=\"/favicon.ico\" /> \n" +
                "<link rel=\"stylesheet\" type=\"text/css\" href=\"/styles/themes/default/easyui.css\">\n" +
                "<link rel=\"stylesheet\" type=\"text/css\" href=\"/styles/themes/icon.css\">\n" +
                "<link rel=\"stylesheet\" type=\"text/css\" href=\"/styles/themes/js_input.css\">\n" +
                "<link rel=\"stylesheet\" href=\"/styles/js/poshytip-1.2/tip-yellowsimple/tip-yellowsimple.css\" type=\"text/css\" />\n" +
                "<link rel=\"stylesheet\" type=\"text/css\" href=\"/styles/themes/main.css\">\n" +
                "<script type=\"text/javascript\" src=\"/styles/js/jquery-1.8.0.min.js\"></script>\n" +
                "<script type=\"text/javascript\" src=\"/styles/js/jquery.easyui.min.js\"></script>\n" +
                "<script type=\"text/javascript\" src=\"/styles/js/jquery.parser.js\"></script>\n" +
                "<script type=\"text/javascript\" src=\"/styles/js/easyui-lang-zh_CN.js\"></script>\n" +
                "<script type=\"text/javascript\" src=\"/styles/layer/layer.min.js\"></script>\n" +
                "<script type=\"text/javascript\" src=\"/styles/js/ntss.js?v=1.5\"></script>\n" +
                "<script type=\"text/javascript\" src=\"/styles/js/js_input.js\"></script>\n" +
                "<script type=\"text/javascript\" src=\"/styles/js/poshytip-1.2/jquery.poshytip.min.js\"></script>\n" +
                "<script type=\"text/javascript\" src=\"/styles/js/entss.js?v=1.2\"></script>\n" +
                "\n" +
                "<style type=\"text/css\">\n" +
                "body{padding:0;}\n" +
                "a{text-decoration:none;color:#1f3a87;text-decoration:none;}\n" +
                "</style>\n" +
                "</head>\n" +
                "<body>\n" +
                "<table class=\"kb\" cellspacing=\"0\" cellpadding=\"0\" border=\"0\">\n" +
                "\t<tr ><th></th><th>一</th><th>二</th><th>三</th><th>四</th><th>五</th><th>六</th><th>日</th></tr>\n" +
                "\t\n" +
                "\t<tr><td style=\"width:60px;padding-left:10px;background-color: #CAE8EA;\">第01节</td><td id=\"01-1\"><div class=\"content\"></div></td><td id=\"01-2\"><div class=\"content\"></div></td><td id=\"01-3\"><div class=\"content\"></div></td><td id=\"01-4\"><div class=\"content\"></div></td><td id=\"01-5\"><div class=\"content\"></div></td><td id=\"01-6\"><div class=\"content\"></div></td><td id=\"01-7\"><div class=\"content\"></div></td></tr>\n" +
                "\t\n" +
                "\t<tr><td style=\"width:60px;padding-left:10px;background-color: #CAE8EA;\">第02节</td><td id=\"02-1\"><div class=\"content\"></div></td><td id=\"02-2\"><div class=\"content\"></div></td><td id=\"02-3\"><div class=\"content\"></div></td><td id=\"02-4\"><div class=\"content\"></div></td><td id=\"02-5\"><div class=\"content\"></div></td><td id=\"02-6\"><div class=\"content\"></div></td><td id=\"02-7\"><div class=\"content\"></div></td></tr>\n" +
                "\t\n" +
                "\t<tr><td style=\"width:60px;padding-left:10px;background-color: #CAE8EA;\">第03节</td><td id=\"03-1\"><div class=\"content\"></div></td><td id=\"03-2\"><div class=\"content\"></div></td><td id=\"03-3\"><div class=\"content\"></div></td><td id=\"03-4\"><div class=\"content\"></div></td><td id=\"03-5\"><div class=\"content\"></div></td><td id=\"03-6\"><div class=\"content\"></div></td><td id=\"03-7\"><div class=\"content\"></div></td></tr>\n" +
                "\t\n" +
                "\t<tr><td style=\"width:60px;padding-left:10px;background-color: #CAE8EA;\">第04节</td><td id=\"04-1\"><div class=\"content\"></div></td><td id=\"04-2\"><div class=\"content\"></div></td><td id=\"04-3\"><div class=\"content\"></div></td><td id=\"04-4\"><div class=\"content\"></div></td><td id=\"04-5\"><div class=\"content\"></div></td><td id=\"04-6\"><div class=\"content\"></div></td><td id=\"04-7\"><div class=\"content\"></div></td></tr>\n" +
                "\t\n" +
                "\t<tr><td style=\"width:60px;padding-left:10px;background-color: #CAE8EA;\">第05节</td><td id=\"05-1\"><div class=\"content\"></div></td><td id=\"05-2\"><div class=\"content\"></div></td><td id=\"05-3\"><div class=\"content\"></div></td><td id=\"05-4\"><div class=\"content\"></div></td><td id=\"05-5\"><div class=\"content\"></div></td><td id=\"05-6\"><div class=\"content\"></div></td><td id=\"05-7\"><div class=\"content\"></div></td></tr>\n" +
                "\t\n" +
                "\t<tr><td style=\"width:60px;padding-left:10px;background-color: #CAE8EA;\">第06节</td><td id=\"06-1\"><div class=\"content\"></div></td><td id=\"06-2\"><div class=\"content\"></div></td><td id=\"06-3\"><div class=\"content\"></div></td><td id=\"06-4\"><div class=\"content\"></div></td><td id=\"06-5\"><div class=\"content\"></div></td><td id=\"06-6\"><div class=\"content\"></div></td><td id=\"06-7\"><div class=\"content\"></div></td></tr>\n" +
                "\t\n" +
                "\t<tr><td style=\"width:60px;padding-left:10px;background-color: #CAE8EA;\">第07节</td><td id=\"07-1\"><div class=\"content\"></div></td><td id=\"07-2\"><div class=\"content\"></div></td><td id=\"07-3\"><div class=\"content\"></div></td><td id=\"07-4\"><div class=\"content\"></div></td><td id=\"07-5\"><div class=\"content\"></div></td><td id=\"07-6\"><div class=\"content\"></div></td><td id=\"07-7\"><div class=\"content\"></div></td></tr>\n" +
                "\t\n" +
                "\t<tr><td style=\"width:60px;padding-left:10px;background-color: #CAE8EA;\">第08节</td><td id=\"08-1\"><div class=\"content\"></div></td><td id=\"08-2\"><div class=\"content\"></div></td><td id=\"08-3\"><div class=\"content\"></div></td><td id=\"08-4\"><div class=\"content\"></div></td><td id=\"08-5\"><div class=\"content\"></div></td><td id=\"08-6\"><div class=\"content\"></div></td><td id=\"08-7\"><div class=\"content\"></div></td></tr>\n" +
                "\t\n" +
                "\t<tr><td style=\"width:60px;padding-left:10px;background-color: #CAE8EA;\">第09节</td><td id=\"09-1\"><div class=\"content\"></div></td><td id=\"09-2\"><div class=\"content\"></div></td><td id=\"09-3\"><div class=\"content\"></div></td><td id=\"09-4\"><div class=\"content\"></div></td><td id=\"09-5\"><div class=\"content\"></div></td><td id=\"09-6\"><div class=\"content\"></div></td><td id=\"09-7\"><div class=\"content\"></div></td></tr>\n" +
                "\t\n" +
                "\t<tr><td style=\"width:60px;padding-left:10px;background-color: #CAE8EA;\">第10节</td><td id=\"10-1\"><div class=\"content\"></div></td><td id=\"10-2\"><div class=\"content\"></div></td><td id=\"10-3\"><div class=\"content\"></div></td><td id=\"10-4\"><div class=\"content\"></div></td><td id=\"10-5\"><div class=\"content\"></div></td><td id=\"10-6\"><div class=\"content\"></div></td><td id=\"10-7\"><div class=\"content\"></div></td></tr>\n" +
                "\t\n" +
                "\t<tr><td style=\"width:60px;padding-left:10px;background-color: #CAE8EA;\">第11节</td><td id=\"11-1\"><div class=\"content\"></div></td><td id=\"11-2\"><div class=\"content\"></div></td><td id=\"11-3\"><div class=\"content\"></div></td><td id=\"11-4\"><div class=\"content\"></div></td><td id=\"11-5\"><div class=\"content\"></div></td><td id=\"11-6\"><div class=\"content\"></div></td><td id=\"11-7\"><div class=\"content\"></div></td></tr>\n" +
                "\t\n" +
                "\t<tr><td style=\"width:60px;padding-left:10px;background-color: #CAE8EA;\">第12节</td><td id=\"12-1\"><div class=\"content\"></div></td><td id=\"12-2\"><div class=\"content\"></div></td><td id=\"12-3\"><div class=\"content\"></div></td><td id=\"12-4\"><div class=\"content\"></div></td><td id=\"12-5\"><div class=\"content\"></div></td><td id=\"12-6\"><div class=\"content\"></div></td><td id=\"12-7\"><div class=\"content\"></div></td></tr>\n" +
                "\t\n" +
                "\t<tr><td style=\"width:60px;padding-left:10px;background-color: #CAE8EA;\">第13节</td><td id=\"13-1\"><div class=\"content\"></div></td><td id=\"13-2\"><div class=\"content\"></div></td><td id=\"13-3\"><div class=\"content\"></div></td><td id=\"13-4\"><div class=\"content\"></div></td><td id=\"13-5\"><div class=\"content\"></div></td><td id=\"13-6\"><div class=\"content\"></div></td><td id=\"13-7\"><div class=\"content\"></div></td></tr>\n" +
                "\t\n" +
                "\t<tr><td style=\"width:60px;padding-left:10px;background-color: #CAE8EA;\">第14节</td><td id=\"14-1\"><div class=\"content\"></div></td><td id=\"14-2\"><div class=\"content\"></div></td><td id=\"14-3\"><div class=\"content\"></div></td><td id=\"14-4\"><div class=\"content\"></div></td><td id=\"14-5\"><div class=\"content\"></div></td><td id=\"14-6\"><div class=\"content\"></div></td><td id=\"14-7\"><div class=\"content\"></div></td></tr>\n" +
                "\t\n" +
                "</table>\n" +
                "\n" +
                "<div style=\"width:1020px;font-weight:600;color:grey;\"></div>\n" +
                "</body>\n" +
                "<script type=\"text/javascript\">\n" +
                "\$(document).ready(function(){\n" +
                "\tshowMask();\n" +
                "\tvar kbxx = [{\"kcmc\":\"毛泽东思想、邓小平理论和三个代表重要思想概论\",\"kcbh\":\"TMP3550\",\"jxbmc\":\"计算机17(1),计算机17(2),计算机17(3),计算机17(4)\",\"kcrwdm\":\"1133131\",\"jcdm2\":\"08,09\",\"zcs\":\"1,16,15,14,13,12,10,9,8,7,6,4,3,2\",\"xq\":\"1\",\"jxcdmcs\":\"教3-109\",\"teaxms\":\"程洁如\"},{\"kcmc\":\"毛泽东思想、邓小平理论和三个代表重要思想概论\",\"kcbh\":\"TMP3550\",\"jxbmc\":\"计算机17(1),计算机17(2),计算机17(3),计算机17(4)\",\"kcrwdm\":\"1133131\",\"jcdm2\":\"08,09\",\"zcs\":\"1,16,15,14,13,12,10,9,8,7,6,4,3,2\",\"xq\":\"3\",\"jxcdmcs\":\"教3-109\",\"teaxms\":\"程洁如\"},{\"kcmc\":\"毛泽东思想、邓小平理论和三个代表重要思想概论\",\"kcbh\":\"TMP3550\",\"jxbmc\":\"计算机17(1),计算机17(2),计算机17(3),计算机17(4)\",\"kcrwdm\":\"1133131\",\"jcdm2\":\"08,09\",\"zcs\":\"1,14,13,12,10,9,8,7,6,4,3,2\",\"xq\":\"4\",\"jxcdmcs\":\"教3-109\",\"teaxms\":\"程洁如\"},{\"kcmc\":\"数据库系统\",\"kcbh\":\"TMP4909\",\"jxbmc\":\"计算机17(3),计算机17(4)\",\"kcrwdm\":\"1133140\",\"jcdm2\":\"01,02\",\"zcs\":\"1,5,10,8,6,4,2,9,7,3\",\"xq\":\"1\",\"jxcdmcs\":\"教5-304\",\"teaxms\":\"谢锐\"},{\"kcmc\":\"数据库系统\",\"kcbh\":\"TMP4909\",\"jxbmc\":\"计算机17(3),计算机17(4)\",\"kcrwdm\":\"1133140\",\"jcdm2\":\"01,02\",\"zcs\":\"4,5,15,14,12,10,8,3,2,1,16,13,11,7,6\",\"xq\":\"5\",\"jxcdmcs\":\"教2-418\",\"teaxms\":\"谢锐\"},{\"kcmc\":\"人工智能\",\"kcbh\":\"TMP4072\",\"jxbmc\":\"计算机17(3),计算机17(4)\",\"kcrwdm\":\"1133148\",\"jcdm2\":\"03,04\",\"zcs\":\"1,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2\",\"xq\":\"1\",\"jxcdmcs\":\"教4-302\",\"teaxms\":\"汪明慧\"},{\"kcmc\":\"大数据技术基础\",\"kcbh\":\"TMP0897\",\"jxbmc\":\"计算机17(1)[大数据技术方向],计算机17(3)[大数据技术方向],计算机17(4)[大数据技术方向],计算机17(2)[大数据技术方向]\",\"kcrwdm\":\"1133193\",\"jcdm2\":\"03,04\",\"zcs\":\"1,6,4,2,13,12,11,8,10,9,7,3\",\"xq\":\"4\",\"jxcdmcs\":\"教3-210\",\"teaxms\":\"张灵\"},{\"kcmc\":\"计算机系统结构\",\"kcbh\":\"TMP2771\",\"jxbmc\":\"计算机17(3),计算机17(4)\",\"kcrwdm\":\"1133210\",\"jcdm2\":\"06,07\",\"zcs\":\"1,11,8,7,3,10,9,6,4,2\",\"xq\":\"1\",\"jxcdmcs\":\"教1-435\",\"teaxms\":\"胡志斌\"},{\"kcmc\":\"计算机系统结构\",\"kcbh\":\"TMP2771\",\"jxbmc\":\"计算机17(3),计算机17(4)\",\"kcrwdm\":\"1133210\",\"jcdm2\":\"01,02\",\"zcs\":\"4,11,8,7,3,2,1,10,9,6\",\"xq\":\"3\",\"jxcdmcs\":\"教3-304\",\"teaxms\":\"胡志斌\"},{\"kcmc\":\"软件工程\",\"kcbh\":\"TMP4226\",\"jxbmc\":\"计算机17(3),计算机17(4)\",\"kcrwdm\":\"1133218\",\"jcdm2\":\"06,07\",\"zcs\":\"3,16,15,14,13,12,11,10,9,8,7,2,1,6,4\",\"xq\":\"3\",\"jxcdmcs\":\"教2-221\",\"teaxms\":\"杨劲涛\"},{\"kcmc\":\"软件工程\",\"kcbh\":\"TMP4226\",\"jxbmc\":\"计算机17(3),计算机17(4)\",\"kcrwdm\":\"1133218\",\"jcdm2\":\"06,07\",\"zcs\":\"3,6,2,1,4\",\"xq\":\"5\",\"jxcdmcs\":\"教1-426\",\"teaxms\":\"杨劲涛\"},{\"kcmc\":\"操作系统\",\"kcbh\":\"TMP0596\",\"jxbmc\":\"计算机17(3),计算机17(4)\",\"kcrwdm\":\"1133226\",\"jcdm2\":\"01,02\",\"zcs\":\"1,7,16,15,14,13,12,11,10,9,8,6,4,3,2\",\"xq\":\"2\",\"jxcdmcs\":\"教3-302\",\"teaxms\":\"申建芳\"},{\"kcmc\":\"操作系统\",\"kcbh\":\"TMP0596\",\"jxbmc\":\"计算机17(3),计算机17(4)\",\"kcrwdm\":\"1133226\",\"jcdm2\":\"03,04\",\"zcs\":\"1,10,9,8,7,6,4,3,2\",\"xq\":\"3\",\"jxcdmcs\":\"教3-303\",\"teaxms\":\"申建芳\"},{\"kcmc\":\"数据挖掘\",\"kcbh\":\"TMP4925\",\"jxbmc\":\"计算机17(1),计算机17(3),计算机17(4),计算机17(2)\",\"kcrwdm\":\"1133233\",\"jcdm2\":\"06,07\",\"zcs\":\"5\",\"xq\":\"1\",\"jxcdmcs\":\"教3-312\",\"teaxms\":\"张巍\"},{\"kcmc\":\"数据挖掘\",\"kcbh\":\"TMP4925\",\"jxbmc\":\"计算机17(1),计算机17(3),计算机17(4),计算机17(2)\",\"kcrwdm\":\"1133233\",\"jcdm2\":\"03,04\",\"zcs\":\"1,16,15,14,11,10,13,12,8,7,6,5,4,3,2\",\"xq\":\"5\",\"jxcdmcs\":\"教3-312\",\"teaxms\":\"张巍\"},{\"kcmc\":\"工程伦理\",\"kcbh\":\"TMP7184\",\"jxbmc\":\"17计算机2-8班\",\"kcrwdm\":\"1134531\",\"jcdm2\":\"05\",\"zcs\":\"2\",\"xq\":\"1\",\"jxcdmcs\":\"3号大教室\",\"teaxms\":\"郑瑞芸\"}];\n" +
                "\t\$.each(kbxx,function(index,item){\n" +
                "\t\tvar kbrow = kbObj(item);\n" +
                "\t\tkbrow.appendSelf();\n" +
                "\t});\n" +
                "\t\$.parser.parse(\$(\"div\"));\n" +
                "\tcloseMask();\n" +
                "});\n" +
                "\n" +
                "\n" +
                "function kbObj(obj){\n" +
                "\tvar _objkb  = new Object();\n" +
                "\t_objkb.jcArr = obj.jcdm2.split(\",\");\n" +
                "\t_objkb.startJc = _objkb.jcArr[0];\n" +
                "\t_objkb.zcSumary = getWeekSummary(obj.zcs);\n" +
                "\t_objkb.kcmc = obj.kcmc.length>7?obj.kcmc.substr(0,7)+\"...\":obj.kcmc;\n" +
                "\t_objkb.tips = \"课程名称：\"+obj.kcmc+\"&#10\"\n" +
                "\t\t\t\t +\"课程编号：\"+obj.kcbh+\"&#10\"\n" +
                "\t\t\t\t +\"周次：\"+_objkb.zcSumary+\"&#10\"\n" +
                "\t\t\t\t +\"授课教师：\"+((obj.teaxms == '')?'未安排':obj.teaxms)+\"&#10\"\n" +
                "\t\t\t\t +\"教学场地：\"+obj.jxcdmcs+\"&#10\"\n" +
                "\t\t\t\t +\"教学班名称：\"+obj.jxbmc;\n" +
                "\t_objkb.appendSelf = function(){\n" +
                "\t\tfor(var i=0;i<_objkb.jcArr.length;i++){\n" +
                "\t\t\tvar _position = _objkb.jcArr[i]+\"-\"+obj.xq;\n" +
                "\t\t\tvar _con = \$(\"#\"+_position+\" .content\");\n" +
                "\t\t\t_con.append(\"<div title=\"+_objkb.tips+\" class='kbdiv' style='float:left;background-color:#AEEEEE;width:132px;height:28px;margin-left:1px;'><a href='javascript:view(\\\"\"+obj.kcrwdm+\"\\\",\\\"\"+obj.kcmc+\"\\\");'>\"+_objkb.kcmc+\"</a>★\"+_objkb.zcSumary+\"<br>\"+obj.jxcdmcs+\"</div>\");\n" +
                "\t\t\tvar len = _con.find(\".kbdiv\").length;\n" +
                "\t\t\tif(len > 1){\n" +
                "\t\t\t\t_con.find(\".kbdiv\").css(\"width\",Math.ceil((132-3*len)/len));\n" +
                "\t\t\t}\n" +
                "\t\t}\n" +
                "\t};\n" +
                "\treturn _objkb;\n" +
                "}\n" +
                "\n" +
                "function view(kcrwdm,kcmc){\n" +
                "\t\$('<div id=\"dlg\"><table id=\"skxxdatalist\"></table></div>').dialog({\n" +
                "\t\twidth:700,height:\$(window).height()-100,\n" +
                "\t\ttitle:kcmc+'上课信息',\n" +
                "\t\threfMode:\"iframe\",modal:true,iconCls:\"icon-edit\",\n" +
                "\t\tonClose:function(){\$(this).dialog(\"destroy\");}\n" +
                "\t});\n" +
                "\t\n" +
                "\t//渲染datalist\n" +
                "\t\$('#skxxdatalist').datagrid({\n" +
                "\t\twidth: 684,\n" +
                "\t\theight: \$(window).height()-136,\n" +
                "\t\tstriped: true,\n" +
                "\t\tsingleSelect: false,\n" +
                "\t\turl:'xsgrkbcx!getSkxxDataList.action',\n" +
                "\t\tpagination: true,\n" +
                "\t\trownumbers: false,\n" +
                "\t\tqueryParams:{\n" +
                "\t\t\t\t\t\tkcrwdm:kcrwdm,\n" +
                "\t\t\t\t\t\tteadm :''\n" +
                "\t\t\t\t\t},\n" +
                "\t\tpageSize: '50',\n" +
                "\t\tpageList:['50','50'*2,'50'*3],\n" +
                "\t\tfitColumns: true,\n" +
                "\t\tsortName: 'kxh',\n" +
                "\t\tcolumns:[[\n" +
                "\t\t\t  {field:'kxh',title:'课序号',width:30,sortable:true},\n" +
                "\t\t\t  {field:'zc',title:'周次',width:30,sortable:true},\n" +
                "\t\t      {field:'xq',title:'星期',width:30,sortable:true},\n" +
                "\t          {field:'jcdm2',title:'节次',width:45,sortable:true},\n" +
                "\t          {field:'jxcdmc',title:'教学场地',width:50,sortable:true},\n" +
                "\t\t      {field:'jxbmc',title:'教学班',width:90,sortable:true},\n" +
                "\t\t      //{field:'kcmc',title:'课程名称',width:90,sortable:true},\n" +
                "\t          {field:'sknrjj',title:'授课内容',width:125,sortable:true}\n" +
                "\t    ]]\n" +
                "\t});\n" +
                "}\n" +
                "</script>\n" +
                "</html>\n" +
                "\n"
        val str = Regex("(?<=var kbxx = )\\[.*]").find(content)?.value
        println(str)
    }

    @Test
    fun termcode_for_schedule_test(){
        val content = "\n" +
                "\n" +
                "\n" +
                "\n" +
                "\n" +
                "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\" \"http://www.w3.org/TR/html4/loose.dtd\">\n" +
                "<html>\n" +
                "<head>\n" +
                "<meta http-equiv=\"pragma\" content=\"no-cache\" /> \n" +
                "<meta http-equiv=\"cache-control\" content=\"no-cache\" /> \n" +
                "<meta http-equiv=\"expires\" content=\"0\" /> \n" +
                "<meta http-equiv=\"Content-Type\" content=\"text/html; charset=UTF-8\" /> \n" +
                "<title>学生个人课表</title>\n" +
                "<link rel=\"stylesheet\" type=\"text/css\" href=\"/styles/themes/default/easyui.css\">\n" +
                "<link rel='stylesheet' href='/styles/fullcalendar/cupertino/theme.css' />\n" +
                "<link href='/styles/fullcalendar/fullcalendar.css' rel='stylesheet' />\n" +
                "<script type=\"text/javascript\" src=\"/styles/js/jquery-1.8.0.min.js\"></script>\n" +
                "<script type=\"text/javascript\" src=\"/styles/js/jquery.easyui.min.js\"></script>\n" +
                "<script type=\"text/javascript\" src=\"/styles/js/easyui-lang-zh_CN.js\"></script>\n" +
                "<script type=\"text/javascript\" src=\"/styles/layer/layer.min.js\"></script>\n" +
                "<script type=\"text/javascript\" src=\"/styles/js/ntss.js\"></script>\n" +
                "</head>\n" +
                "<body>\n" +
                "\t<table border=\"0\" cellspacing=\"0\" cellpadding=\"0\" height='20px' style=\"width:600px;padding:5px;margin-top:5px;\">\n" +
                "\t\t\t<tr>\t\t\t\t\n" +
                "\t\t\t\t<td align=\"right\" >学期：</td>\n" +
                "\t\t\t\t<td >\n" +
                "\t\t\t\t\t<select id='xnxqdm' name='xnxqdm' style='width:120px' class='ntssselect' ><option value='202402' >2025春季</option><option value='202401' >2024秋季</option><option value='202302' >2024春季</option><option value='202301' >2023秋季</option><option value='202202' >2023春季</option><option value='202201' >2022秋季</option><option value='202102' >2022春季</option><option value='202101' >2021秋季</option><option value='202002' >2021春季</option><option value='202001' >2020秋季</option><option value='201902' >2020春季</option><option value='201901' selected>2019秋季</option><option value='201802' >2019春季</option><option value='201801' >2018秋季</option><option value='201702' >2018春季</option><option value='201701' >2017秋季</option><option value='201602' >2017春季</option><option value='201601' >2016秋季</option><option value='201502' >2016春季</option><option value='201501' >2015秋季</option><option value='201402' >2015春季</option><option value='201401' >2014秋季</option><option value='201302' >2014春季</option><option value='201301' >2013秋季</option><option value='201202' >2013春季</option><option value='201201' >2012秋季</option><option value='201102' >2012春季</option><option value='201101' >2011秋季</option><option value='201002' >2011春季</option><option value='201001' >2010秋季</option></select>\n" +
                "\t\t\t\t</td>\n" +
                "\t\t\t\t<td align=\"right\" >周次：</td>\n" +
                "\t\t\t\t<td>\n" +
                "\t\t\t\t\t<select id=\"zc\" class='ntssselect' style=\"width:80px;\"></select>\n" +
                "\t\t\t\t</td>\n" +
                "\t\t\t\t<td>\n" +
                "\t\t\t\t\t<div class=\"fc-header\" style=\"margin-left:10px;\">\n" +
                "\t\t\t\t\t\t<span id=\"preWeek\" class=\"fc-button fc-button-prev ui-state-default ui-corner-left selfB\">\n" +
                "\t\t\t\t\t\t\t<span class=\"fc-icon-wrap\"><span class=\"ui-icon ui-icon-circle-triangle-w\"></span></span>\n" +
                "\t\t\t\t\t\t</span>\n" +
                "\t\t\t\t\t\t<span id=\"nextWeek\" class=\"fc-button fc-button-next ui-state-default ui-corner-right selfB\">\n" +
                "\t\t\t\t\t\t\t<span class=\"fc-icon-wrap\"><span class=\"ui-icon ui-icon-circle-triangle-e\"></span></span>\n" +
                "\t\t\t\t\t\t</span>\n" +
                "\t\t\t\t\t</div>\n" +
                "\t\t\t\t</td>\n" +
                "\t       \t\t<td>\n" +
                "\t\t\t\t\t<div class=\"fc-header\" style=\"margin-left:10px;\">\n" +
                "\t\t\t\t\t<span class=\"fc-button fc-button-today ui-state-default ui-corner-left ui-corner-right selfB\" id=\"bfind\">查询课表</span>\n" +
                "\t\t\t\t\t<span class=\"fc-button fc-button-today ui-state-default ui-corner-left ui-corner-right selfB\" id=\"blist\">列表展示</span>\n" +
                "\t\t\t\t\t<!--  <span class=\"fc-button fc-button-today ui-state-default ui-corner-left ui-corner-right selfB\" id=\"exportkb\" data-printType=\"teakbPrint\">打印</span>-->\n" +
                "\t\t\t\t\t</div>\n" +
                "\t       \t\t</td>\n" +
                "\t\t\t\t<td colspan=\"2\"></td>\n" +
                "\t\t\t</tr>\n" +
                "\t</table>\n" +
                "\t<iframe id=\"list\" width=\"100%\" frameborder='no' border='0' src='' scrolling=\"yes\"></iframe>\n" +
                "<div id=\"poplist\" style=\"width:0;height:0;\">\n" +
                "    <iframe scrolling=\"auto\" id='frmlist' frameborder=\"0\"  src=\"\" style=\"width:100%;height:100%;\"></iframe>\n" +
                "</div>\n" +
                "</body>\n" +
                "<script>\n" +
                "\$(document).ready(function(){\n" +
                "\tvar \$list = \$(\"#list\");\n" +
                "\tif('' == 'false'){\n" +
                "\t\t\$.messager.alert('信息','本学期课表还未发布，请稍后关注!','warning');\n" +
                "\t}\n" +
                "\t\$list.height(\$(document).height()-50);\n" +
                "\t//初始化周次下拉框\n" +
                "\tvar \$zc = \$('#zc');\n" +
                "\t\$zc.append(\"<option value='' >全部</option>\");\n" +
                "\tfor (var i = 0; i < 22; i++) {\n" +
                "\t\t\$zc.append(\"<option value='\"+(i+1)+\"' >第\"+(i+1)+\"周</option>\");\n" +
                "\t}\n" +
                "\tvar val = '' || '' || \"\";\n" +
                "\t\$zc.val(val);\n" +
                "\t\n" +
                "\t\$(\"#bfind\").click(function(){\n" +
                "\t\tif(\$zc.val()=='')\n" +
                "\t\t\t \$list.attr('src', 'xsgrkbcx!xsAllKbList.action?xnxqdm='+\$(\"#xnxqdm\").val());\n" +
                "\t\telse\n" +
                "\t\t\t \$list.attr('src', 'xsgrkbcx!xskbList.action?xnxqdm='+\$(\"#xnxqdm\").val()+'&zc='+\$zc.val());\n" +
                "\t\n" +
                "\t});\t\n" +
                "\t\n" +
                "\t\n" +
                "\t//上一周，下一周\t\t\n" +
                "\t\$('#preWeek').click(function() {\n" +
                "\t\tif (\$zc.val() == \"\") {\n" +
                "\t\t\t\$zc.val(22);//最后一周\n" +
                "\t\t\t\$list.attr('src', 'xsgrkbcx!xskbList.action?xnxqdm='+\$(\"#xnxqdm\").val()\n" +
                "\t\t\t\t\t\t\t\t\t\t\t\t\t+'&zc='+22);\n" +
                "\t\t}else if(\$zc.val()==1){//第一周\n" +
                "\t\t\t\$zc.val('');//全部周\n" +
                "\t\t\t\$list.attr('src', 'xsgrkbcx!xsAllKbList.action?xnxqdm='+\$(\"#xnxqdm\").val());\n" +
                "\t\t}else {\n" +
                "\t\t\tvar djz=parseInt(\$zc.val(),10)-1;\n" +
                "\t\t\t\$zc.val(djz);\n" +
                "\t\t\t\$list.attr('src', 'xsgrkbcx!xskbList.action?xnxqdm='+\$(\"#xnxqdm\").val()\n" +
                "\t\t\t\t\t\t\t\t\t\t\t\t\t+'&zc='+djz);\n" +
                "\t\t}\n" +
                "\t});\n" +
                "\t\$('#nextWeek').click(function() {\n" +
                "\t\tif (\$zc.val() == 22) {\n" +
                "\t\t\t\$zc.val(\"\");\n" +
                "\t\t\t\$list.attr('src', 'xsgrkbcx!xsAllKbList.action?xnxqdm='+\$(\"#xnxqdm\").val());\n" +
                "\t\t}else if (\$zc.val() == \"\") {\n" +
                "\t\t\t\$zc.val(1);//第一周\n" +
                "\t\t\t\$list.attr('src', 'xsgrkbcx!xskbList.action?xnxqdm='+\$(\"#xnxqdm\").val()\n" +
                "\t\t\t\t\t\t\t\t\t\t\t\t\t+'&zc='+1);\n" +
                "\t\t}else {\n" +
                "\t\t\tvar djz=parseInt(\$zc.val(),10)+1;\n" +
                "\t\t\t\$zc.val(djz);\t\n" +
                "\t\t\t\$list.attr('src', 'xsgrkbcx!xskbList.action?xnxqdm='+\$(\"#xnxqdm\").val()\n" +
                "\t\t\t\t\t\t\t\t\t\t\t\t\t+'&zc='+djz);\n" +
                "\t\t}\n" +
                "\t});\n" +
                "\t\n" +
                "\t\$(\"#blist\").click(function(){\n" +
                "\t\t\$('#frmlist').attr('src','xsgrkbcx!xskbList2.action?xnxqdm='+\$(\"#xnxqdm\").val()+'&zc='+\$(\"#zc\").val());\n" +
                "\t\t\$('#poplist').dialog({\n" +
                "\t\t\twidth:850,\n" +
                "\t\t\theight:500,\n" +
                "\t\t\ttitle:\"学生课表信息\",\n" +
                "\t\t\tmodal:true,\n" +
                "\t\t\threfMode:\"iframe\"\n" +
                "\t\t});\n" +
                "\t});\n" +
                "\n" +
                "\t\$('body').on('mouseover','.selfB',function(){\$(this).addClass('ui-state-hover');});\n" +
                "\t\$('body').on('mouseout','.selfB',function(){\$(this).removeClass('ui-state-hover');});\n" +
                "\t\n" +
                "\t\$(\"#bfind\").click();\n" +
                "});\n" +
                "\n" +
                "function doChangeXnxq(){\n" +
                "\twindow.location=\"xsgrkbcx!xsgrkbMain.action?xnxqdm=\"+\$(\"#xnxqdm\").val();\n" +
                "}\n" +
                "</script>\n" +
                "</html>"
        val result = Regex("(?<=<option value=')\\d{6}(?=' selected>)").find(content)?.value?:""
        println(result)
    }

    @Test
    fun split_test(){
        val str = "1,7,16,15,14,13,12,11,10,9,8,6,4,3,2"
        println(str.split(",")[0])
    }

    @Test
    fun time_test(){
        val cal: Calendar = Calendar.getInstance()
        val month: Int = cal.get(Calendar.MONTH) + 1
        val year: Int = cal.get(Calendar.YEAR)
        print("$year, $month")
    }

}