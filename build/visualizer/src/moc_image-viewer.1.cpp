/****************************************************************************
** Meta object code from reading C++ file 'image-viewer.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.15.3)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include <memory>
#include "../../../visualizer/src/image-viewer.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'image-viewer.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.15.3. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_pal__ImageViewer_t {
    QByteArrayData data[25];
    char stringdata0[228];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_pal__ImageViewer_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_pal__ImageViewer_t qt_meta_stringdata_pal__ImageViewer = {
    {
QT_MOC_LITERAL(0, 0, 16), // "pal::ImageViewer"
QT_MOC_LITERAL(1, 17, 12), // "imageChanged"
QT_MOC_LITERAL(2, 30, 0), // ""
QT_MOC_LITERAL(3, 31, 11), // "zoomChanged"
QT_MOC_LITERAL(4, 43, 5), // "scale"
QT_MOC_LITERAL(5, 49, 7), // "setText"
QT_MOC_LITERAL(6, 57, 3), // "txt"
QT_MOC_LITERAL(7, 61, 8), // "setImage"
QT_MOC_LITERAL(8, 70, 11), // "setRotation"
QT_MOC_LITERAL(9, 82, 5), // "angle"
QT_MOC_LITERAL(10, 88, 18), // "setAspectRatioMode"
QT_MOC_LITERAL(11, 107, 19), // "Qt::AspectRatioMode"
QT_MOC_LITERAL(12, 127, 17), // "aspect_ratio_mode"
QT_MOC_LITERAL(13, 145, 7), // "zoomFit"
QT_MOC_LITERAL(14, 153, 12), // "zoomOriginal"
QT_MOC_LITERAL(15, 166, 6), // "zoomIn"
QT_MOC_LITERAL(16, 173, 5), // "level"
QT_MOC_LITERAL(17, 179, 7), // "zoomOut"
QT_MOC_LITERAL(18, 187, 7), // "mouseAt"
QT_MOC_LITERAL(19, 195, 1), // "x"
QT_MOC_LITERAL(20, 197, 1), // "y"
QT_MOC_LITERAL(21, 199, 15), // "updateSceneRect"
QT_MOC_LITERAL(22, 215, 1), // "w"
QT_MOC_LITERAL(23, 217, 1), // "h"
QT_MOC_LITERAL(24, 219, 8) // "rotation"

    },
    "pal::ImageViewer\0imageChanged\0\0"
    "zoomChanged\0scale\0setText\0txt\0setImage\0"
    "setRotation\0angle\0setAspectRatioMode\0"
    "Qt::AspectRatioMode\0aspect_ratio_mode\0"
    "zoomFit\0zoomOriginal\0zoomIn\0level\0"
    "zoomOut\0mouseAt\0x\0y\0updateSceneRect\0"
    "w\0h\0rotation"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_pal__ImageViewer[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
      14,   14, // methods
       1,  120, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       2,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    0,   84,    2, 0x06 /* Public */,
       3,    1,   85,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       5,    1,   88,    2, 0x0a /* Public */,
       7,    1,   91,    2, 0x0a /* Public */,
       8,    1,   94,    2, 0x0a /* Public */,
      10,    1,   97,    2, 0x0a /* Public */,
      13,    0,  100,    2, 0x0a /* Public */,
      14,    0,  101,    2, 0x0a /* Public */,
      15,    1,  102,    2, 0x0a /* Public */,
      15,    0,  105,    2, 0x2a /* Public | MethodCloned */,
      17,    1,  106,    2, 0x0a /* Public */,
      17,    0,  109,    2, 0x2a /* Public | MethodCloned */,
      18,    2,  110,    2, 0x0a /* Public */,
      21,    2,  115,    2, 0x08 /* Private */,

 // signals: parameters
    QMetaType::Void,
    QMetaType::Void, QMetaType::Double,    4,

 // slots: parameters
    QMetaType::Void, QMetaType::QString,    6,
    QMetaType::Void, QMetaType::QImage,    2,
    QMetaType::Void, QMetaType::QReal,    9,
    QMetaType::Void, 0x80000000 | 11,   12,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int,   16,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int,   16,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int, QMetaType::Int,   19,   20,
    QMetaType::Void, QMetaType::Int, QMetaType::Int,   22,   23,

 // properties: name, type, flags
      24, QMetaType::QReal, 0x00095103,

       0        // eod
};

void pal::ImageViewer::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<ImageViewer *>(_o);
        (void)_t;
        switch (_id) {
        case 0: _t->imageChanged(); break;
        case 1: _t->zoomChanged((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 2: _t->setText((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        case 3: _t->setImage((*reinterpret_cast< const QImage(*)>(_a[1]))); break;
        case 4: _t->setRotation((*reinterpret_cast< qreal(*)>(_a[1]))); break;
        case 5: _t->setAspectRatioMode((*reinterpret_cast< Qt::AspectRatioMode(*)>(_a[1]))); break;
        case 6: _t->zoomFit(); break;
        case 7: _t->zoomOriginal(); break;
        case 8: _t->zoomIn((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 9: _t->zoomIn(); break;
        case 10: _t->zoomOut((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 11: _t->zoomOut(); break;
        case 12: _t->mouseAt((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 13: _t->updateSceneRect((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (ImageViewer::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&ImageViewer::imageChanged)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (ImageViewer::*)(double );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&ImageViewer::zoomChanged)) {
                *result = 1;
                return;
            }
        }
    }
#ifndef QT_NO_PROPERTIES
    else if (_c == QMetaObject::ReadProperty) {
        auto *_t = static_cast<ImageViewer *>(_o);
        (void)_t;
        void *_v = _a[0];
        switch (_id) {
        case 0: *reinterpret_cast< qreal*>(_v) = _t->rotation(); break;
        default: break;
        }
    } else if (_c == QMetaObject::WriteProperty) {
        auto *_t = static_cast<ImageViewer *>(_o);
        (void)_t;
        void *_v = _a[0];
        switch (_id) {
        case 0: _t->setRotation(*reinterpret_cast< qreal*>(_v)); break;
        default: break;
        }
    } else if (_c == QMetaObject::ResetProperty) {
    }
#endif // QT_NO_PROPERTIES
}

QT_INIT_METAOBJECT const QMetaObject pal::ImageViewer::staticMetaObject = { {
    QMetaObject::SuperData::link<QFrame::staticMetaObject>(),
    qt_meta_stringdata_pal__ImageViewer.data,
    qt_meta_data_pal__ImageViewer,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *pal::ImageViewer::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *pal::ImageViewer::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_pal__ImageViewer.stringdata0))
        return static_cast<void*>(this);
    return QFrame::qt_metacast(_clname);
}

int pal::ImageViewer::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QFrame::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 14)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 14;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 14)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 14;
    }
#ifndef QT_NO_PROPERTIES
    else if (_c == QMetaObject::ReadProperty || _c == QMetaObject::WriteProperty
            || _c == QMetaObject::ResetProperty || _c == QMetaObject::RegisterPropertyMetaType) {
        qt_static_metacall(this, _c, _id, _a);
        _id -= 1;
    } else if (_c == QMetaObject::QueryPropertyDesignable) {
        _id -= 1;
    } else if (_c == QMetaObject::QueryPropertyScriptable) {
        _id -= 1;
    } else if (_c == QMetaObject::QueryPropertyStored) {
        _id -= 1;
    } else if (_c == QMetaObject::QueryPropertyEditable) {
        _id -= 1;
    } else if (_c == QMetaObject::QueryPropertyUser) {
        _id -= 1;
    }
#endif // QT_NO_PROPERTIES
    return _id;
}

// SIGNAL 0
void pal::ImageViewer::imageChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 0, nullptr);
}

// SIGNAL 1
void pal::ImageViewer::zoomChanged(double _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}
struct qt_meta_stringdata_pal__PixmapItem_t {
    QByteArrayData data[12];
    char stringdata0[87];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_pal__PixmapItem_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_pal__PixmapItem_t qt_meta_stringdata_pal__PixmapItem = {
    {
QT_MOC_LITERAL(0, 0, 15), // "pal::PixmapItem"
QT_MOC_LITERAL(1, 16, 13), // "doubleClicked"
QT_MOC_LITERAL(2, 30, 0), // ""
QT_MOC_LITERAL(3, 31, 1), // "x"
QT_MOC_LITERAL(4, 33, 1), // "y"
QT_MOC_LITERAL(5, 35, 12), // "imageChanged"
QT_MOC_LITERAL(6, 48, 11), // "sizeChanged"
QT_MOC_LITERAL(7, 60, 1), // "w"
QT_MOC_LITERAL(8, 62, 1), // "h"
QT_MOC_LITERAL(9, 64, 10), // "mouseMoved"
QT_MOC_LITERAL(10, 75, 8), // "setImage"
QT_MOC_LITERAL(11, 84, 2) // "im"

    },
    "pal::PixmapItem\0doubleClicked\0\0x\0y\0"
    "imageChanged\0sizeChanged\0w\0h\0mouseMoved\0"
    "setImage\0im"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_pal__PixmapItem[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       5,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       4,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    2,   39,    2, 0x06 /* Public */,
       5,    1,   44,    2, 0x06 /* Public */,
       6,    2,   47,    2, 0x06 /* Public */,
       9,    2,   52,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
      10,    1,   57,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void, QMetaType::Int, QMetaType::Int,    3,    4,
    QMetaType::Void, QMetaType::QImage,    2,
    QMetaType::Void, QMetaType::Int, QMetaType::Int,    7,    8,
    QMetaType::Void, QMetaType::Int, QMetaType::Int,    3,    4,

 // slots: parameters
    QMetaType::Void, QMetaType::QImage,   11,

       0        // eod
};

void pal::PixmapItem::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<PixmapItem *>(_o);
        (void)_t;
        switch (_id) {
        case 0: _t->doubleClicked((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 1: _t->imageChanged((*reinterpret_cast< const QImage(*)>(_a[1]))); break;
        case 2: _t->sizeChanged((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 3: _t->mouseMoved((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 4: _t->setImage((*reinterpret_cast< QImage(*)>(_a[1]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (PixmapItem::*)(int , int );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&PixmapItem::doubleClicked)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (PixmapItem::*)(const QImage & );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&PixmapItem::imageChanged)) {
                *result = 1;
                return;
            }
        }
        {
            using _t = void (PixmapItem::*)(int , int );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&PixmapItem::sizeChanged)) {
                *result = 2;
                return;
            }
        }
        {
            using _t = void (PixmapItem::*)(int , int );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&PixmapItem::mouseMoved)) {
                *result = 3;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject pal::PixmapItem::staticMetaObject = { {
    QMetaObject::SuperData::link<QObject::staticMetaObject>(),
    qt_meta_stringdata_pal__PixmapItem.data,
    qt_meta_data_pal__PixmapItem,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *pal::PixmapItem::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *pal::PixmapItem::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_pal__PixmapItem.stringdata0))
        return static_cast<void*>(this);
    if (!strcmp(_clname, "QGraphicsPixmapItem"))
        return static_cast< QGraphicsPixmapItem*>(this);
    return QObject::qt_metacast(_clname);
}

int pal::PixmapItem::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 5)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 5;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 5)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 5;
    }
    return _id;
}

// SIGNAL 0
void pal::PixmapItem::doubleClicked(int _t1, int _t2)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t2))) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void pal::PixmapItem::imageChanged(const QImage & _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
void pal::PixmapItem::sizeChanged(int _t1, int _t2)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t2))) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}

// SIGNAL 3
void pal::PixmapItem::mouseMoved(int _t1, int _t2)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t2))) };
    QMetaObject::activate(this, &staticMetaObject, 3, _a);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
