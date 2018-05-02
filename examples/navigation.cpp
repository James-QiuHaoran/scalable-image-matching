#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <vector>
using namespace std;

string exec(const char* cmd) {
    array<char, 128> buffer;
    string result;
    shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);
    if (!pipe) throw runtime_error("popen() failed!");
    while (!feof(pipe.get())) {
        if (fgets(buffer.data(), 128, pipe.get()) != nullptr)
            result += buffer.data();
    }
    return result;
}

int main() {
    string target = "/home/jamesqiu/Desktop/test-set2/3_5_9_1.jpg";
    string command = "python first_image.py " + target;
    const char* cmd = command.c_str();
    string result = exec(cmd);
    // cout << result << endl;

    vector<string> paths, batches, directions, signatures;

    int found = result.find("[P]");
    int length = 0;
    while (found != -1) {
        length = result.find("[/P]", found+3) - found - 3;
        string path = result.substr(found+3, length);
        paths.push_back(path);

        found = result.find("[B]", found+3);
        // cout << " - " << found;
        length = result.find("[/B]", found+3) - found - 3;
        string batch = result.substr(found+3, length);
        batches.push_back(batch);

        found = result.find("[D]", found+3);
        // cout << " - " << found;
        length = result.find("[/D]", found+3) - found - 3;
        string direction = result.substr(found+3, length);
        directions.push_back(direction);

        found = result.find("[S]", found+3);
        cout << " - " << found << endl;
        length = result.find("[/S]", found+3) - found - 3;
        string signature = result.substr(found+3, length);
        cout << length << endl;
        signatures.push_back(signature);

        found = result.find("[P]", found+3);
    }

    cout << "# of images: " << paths.size() << endl;
    cout << "# of batches: " << batches.size() << endl;
    cout << "# of directions: " << directions.size() << endl;
    cout << "# of signatures: " << signatures.size() << endl;
    cout << "\nE.g. one of the elements from the parsed string:" << endl;
    cout << "Path: " << paths[0] << endl;
    cout << "Batch #" << batches[0] << " Direction: " << directions[0] << endl;
    cout << "Signature: \n" << signatures[0] << endl;
    cout << "All images: " << endl;
    for (int i = 0; i < paths.size(); i++) {
        cout << paths[i] << " ";
        cout << directions[i] << " ";
        cout << batches[i] << endl;
    }

    string signature1 = signatures[0];
    string signature2 = signatures[1];
    string signature3 = signatures[2];

    // cout << "S1: " << signature1 << endl;
    // cout << "S2: " << signature2 << endl;
    // cout << "S3: " << signature3 << endl;

    command = "python successive_image.py 3 " + target + " \"" + signature1 + "\" \"" + signature2 + "\" \"" + signature3 + "\"";
    const char* cmd2 = command.c_str();
    result = exec(cmd2);
    cout << result << endl;

    return 0;
}

/*
#include <Python.h>
#include <iostream>
using namespace std;

int main() {

    PyObject *pModule=NULL;

    Py_Initialize();
    int ret = Py_IsInitialized();
    if (ret == 0)
        cout << "initialization faied" << endl;
    else
        cout << "initialization succeed" << endl;

    if(!(pModule=PyImport_Import(PyString_FromString("localization_test_scale"))))
    {
        cout<<"get module failed!"<<endl;
        exit(0);
    }

    PyObject* myModuleString = PyUnicode_FromString((char*)"localization_test_scale");
    PyObject* myModule = PyImport_Import(myModuleString);
    if (myModule == NULL)
        cout << "import failed!" << endl;
    else
        cout << "success" << endl;
    // PyObject* myFunctionString = PyUnicode_FromString((char*)"image_match");
    PyObject* myFunction = PyObject_GetAttrString(myModule, (char*)"image_match");
    PyObject* args = PyTuple_Pack(2, 
        PyUnicode_FromString("/home/jamesqiu/Desktop/first_try/3.JPG"), PyFloat_FromDouble(0.6));
    cout << "TEST" << endl;
    PyObject* myResult = PyObject_CallObject(myFunction, args);
    char *result = PyBytes_AS_STRING(myResult);
    // string result = PyString_AsString(myResult);
    cout << result << endl;
    Py_Finalize();
    return 0;
}*/
