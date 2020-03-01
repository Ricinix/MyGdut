//
// Created by laugh on 2020/2/29.
//

#include <jni.h>
#include <string>

using namespace std;

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_mygdut_MyJniClass_stringTest(JNIEnv *env, jobject thiz) {
    char arr[] = "你好a, JNI！";
    return env->NewStringUTF(arr);
//    return str2jstring(env, arr);
}


extern "C"
JNIEXPORT jobject JNICALL
Java_com_example_mygdut_domain_HtmlParser_getHeaders(JNIEnv *env, jobject thiz, jobject instream) {
    // 先获取class
    jclass InputStreamClass = env->FindClass("java/io/InputStream");
    jclass ListClass = env->FindClass("java/util/LinkedList");

    // 获取方法
    jmethodID steamRead = env->GetMethodID(InputStreamClass, "read", "([B)I");
    jmethodID steamClose = env->GetMethodID(InputStreamClass, "close", "()V");
    jmethodID listAdd = env->GetMethodID(ListClass, "add", "(ILjava/lang/Object;)V");
    jmethodID listConstructor = env->GetMethodID(ListClass, "<init>", "()V");

    // 实例化一个List
    jobject myList = env->NewObject(ListClass, listConstructor);

    // 一些变量
    jsize arraySize = 4096;
    jint len;
    int strLen = 4096;
    int strIndex = 0;
    char cBuf[128];
    char *charArray = (char *) malloc(sizeof(char) * strLen);

    bool inHeader = false;

    jbyteArray jBuf = env->NewByteArray(arraySize);

    // 真正的算法
    while ((len = env->CallIntMethod(instream, steamRead, jBuf)) != -1) {
        env->SetByteArrayRegion(jBuf, 0, 4096, (jbyte*)cBuf);
        for (int i = 0; i < len; ++i) {
            char c = cBuf[i];
            if (c == '<') {
                inHeader = true;
                if (strIndex == strLen - 1) {
                    strLen *= 2;
                    charArray = (char *) realloc(charArray, sizeof(char) * strLen);
                }
                charArray[strIndex++] = c;
                continue;
            } else if (c == '\n') continue;
            if (inHeader) {
                if (strIndex == strLen - 1) {
                    strLen *= 2;
                    charArray = (char *) realloc(charArray, sizeof(char) * strLen);
                }
                charArray[strIndex++] = c;
            }
            if (c == '>') {
                inHeader = false;
                charArray[strIndex] = '\0';
                env->CallVoidMethod(myList, listAdd, 0, env->NewStringUTF(string(charArray, strIndex)));
                strIndex = 0;
            }
        }

    }

    env->CallVoidMethod(instream, steamClose);
    env->DeleteLocalRef(thiz);
    env->DeleteLocalRef(instream);
    env->DeleteLocalRef(jBuf);
    free(charArray);
    return myList;
}