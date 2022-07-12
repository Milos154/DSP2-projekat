/****************************************************************************
** Meta object code from reading C++ file 'rect-selection.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.15.3)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include <memory>
#include "../../../visualizer/src/rect-selection.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'rect-selection.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.15.3. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_pal__SelectionItem_t {
    QByteArrayData data[23];
    char stringdata0[265];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_pal__SelectionItem_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_pal__SelectionItem_t qt_meta_stringdata_pal__SelectionItem = {
    {
QT_MOC_LITERAL(0, 0, 18), // "pal::SelectionItem"
QT_MOC_LITERAL(1, 19, 11), // "leftChanged"
QT_MOC_LITERAL(2, 31, 0), // ""
QT_MOC_LITERAL(3, 32, 12), // "rightChanged"
QT_MOC_LITERAL(4, 45, 10), // "topChanged"
QT_MOC_LITERAL(5, 56, 13), // "bottomChanged"
QT_MOC_LITERAL(6, 70, 16), // "selectionChanged"
QT_MOC_LITERAL(7, 87, 7), // "setLeft"
QT_MOC_LITERAL(8, 95, 1), // "p"
QT_MOC_LITERAL(9, 97, 8), // "setRight"
QT_MOC_LITERAL(10, 106, 6), // "setTop"
QT_MOC_LITERAL(11, 113, 9), // "setBottom"
QT_MOC_LITERAL(12, 123, 12), // "setSelection"
QT_MOC_LITERAL(13, 136, 3), // "sel"
QT_MOC_LITERAL(14, 140, 14), // "resetSelection"
QT_MOC_LITERAL(15, 155, 10), // "setVisible"
QT_MOC_LITERAL(16, 166, 15), // "updateSelection"
QT_MOC_LITERAL(17, 182, 19), // "updateSelectionRect"
QT_MOC_LITERAL(18, 202, 20), // "updateSelectionItems"
QT_MOC_LITERAL(19, 223, 9), // "leftMoved"
QT_MOC_LITERAL(20, 233, 10), // "rightMoved"
QT_MOC_LITERAL(21, 244, 8), // "topMoved"
QT_MOC_LITERAL(22, 253, 11) // "bottomMoved"

    },
    "pal::SelectionItem\0leftChanged\0\0"
    "rightChanged\0topChanged\0bottomChanged\0"
    "selectionChanged\0setLeft\0p\0setRight\0"
    "setTop\0setBottom\0setSelection\0sel\0"
    "resetSelection\0setVisible\0updateSelection\0"
    "updateSelectionRect\0updateSelectionItems\0"
    "leftMoved\0rightMoved\0topMoved\0bottomMoved"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_pal__SelectionItem[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
      19,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       5,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,  109,    2, 0x06 /* Public */,
       3,    1,  112,    2, 0x06 /* Public */,
       4,    1,  115,    2, 0x06 /* Public */,
       5,    1,  118,    2, 0x06 /* Public */,
       6,    1,  121,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       7,    1,  124,    2, 0x0a /* Public */,
       9,    1,  127,    2, 0x0a /* Public */,
      10,    1,  130,    2, 0x0a /* Public */,
      11,    1,  133,    2, 0x0a /* Public */,
      12,    1,  136,    2, 0x0a /* Public */,
      14,    0,  139,    2, 0x0a /* Public */,
      15,    1,  140,    2, 0x0a /* Public */,
      16,    1,  143,    2, 0x08 /* Private */,
      17,    0,  146,    2, 0x08 /* Private */,
      18,    1,  147,    2, 0x08 /* Private */,
      19,    1,  150,    2, 0x08 /* Private */,
      20,    1,  153,    2, 0x08 /* Private */,
      21,    1,  156,    2, 0x08 /* Private */,
      22,    1,  159,    2, 0x08 /* Private */,

 // signals: parameters
    QMetaType::Void, QMetaType::Double,    2,
    QMetaType::Void, QMetaType::Double,    2,
    QMetaType::Void, QMetaType::Double,    2,
    QMetaType::Void, QMetaType::Double,    2,
    QMetaType::Void, QMetaType::QRectF,    2,

 // slots: parameters
    QMetaType::Void, QMetaType::Double,    8,
    QMetaType::Void, QMetaType::Double,    8,
    QMetaType::Void, QMetaType::Double,    8,
    QMetaType::Void, QMetaType::Double,    8,
    QMetaType::Void, QMetaType::QRectF,   13,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Bool,    2,
    QMetaType::Void, QMetaType::Int,    2,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int,    2,
    QMetaType::Void, QMetaType::Double,    2,
    QMetaType::Void, QMetaType::Double,    2,
    QMetaType::Void, QMetaType::Double,    2,
    QMetaType::Void, QMetaType::Double,    2,

       0        // eod
};

void pal::SelectionItem::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<SelectionItem *>(_o);
        (void)_t;
        switch (_id) {
        case 0: _t->leftChanged((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 1: _t->rightChanged((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 2: _t->topChanged((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 3: _t->bottomChanged((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 4: _t->selectionChanged((*reinterpret_cast< const QRectF(*)>(_a[1]))); break;
        case 5: _t->setLeft((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 6: _t->setRight((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 7: _t->setTop((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 8: _t->setBottom((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 9: _t->setSelection((*reinterpret_cast< const QRectF(*)>(_a[1]))); break;
        case 10: _t->resetSelection(); break;
        case 11: _t->setVisible((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 12: _t->updateSelection((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 13: _t->updateSelectionRect(); break;
        case 14: _t->updateSelectionItems((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 15: _t->leftMoved((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 16: _t->rightMoved((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 17: _t->topMoved((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 18: _t->bottomMoved((*reinterpret_cast< double(*)>(_a[1]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (SelectionItem::*)(double );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&SelectionItem::leftChanged)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (SelectionItem::*)(double );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&SelectionItem::rightChanged)) {
                *result = 1;
                return;
            }
        }
        {
            using _t = void (SelectionItem::*)(double );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&SelectionItem::topChanged)) {
                *result = 2;
                return;
            }
        }
        {
            using _t = void (SelectionItem::*)(double );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&SelectionItem::bottomChanged)) {
                *result = 3;
                return;
            }
        }
        {
            using _t = void (SelectionItem::*)(const QRectF & );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&SelectionItem::selectionChanged)) {
                *result = 4;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject pal::SelectionItem::staticMetaObject = { {
    QMetaObject::SuperData::link<QGraphicsObject::staticMetaObject>(),
    qt_meta_stringdata_pal__SelectionItem.data,
    qt_meta_data_pal__SelectionItem,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *pal::SelectionItem::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *pal::SelectionItem::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_pal__SelectionItem.stringdata0))
        return static_cast<void*>(this);
    return QGraphicsObject::qt_metacast(_clname);
}

int pal::SelectionItem::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QGraphicsObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 19)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 19;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 19)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 19;
    }
    return _id;
}

// SIGNAL 0
void pal::SelectionItem::leftChanged(double _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void pal::SelectionItem::rightChanged(double _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
void pal::SelectionItem::topChanged(double _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}

// SIGNAL 3
void pal::SelectionItem::bottomChanged(double _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 3, _a);
}

// SIGNAL 4
void pal::SelectionItem::selectionChanged(const QRectF & _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 4, _a);
}
struct qt_meta_stringdata_pal__RectHandle_t {
    QByteArrayData data[10];
    char stringdata0[88];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_pal__RectHandle_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_pal__RectHandle_t qt_meta_stringdata_pal__RectHandle = {
    {
QT_MOC_LITERAL(0, 0, 15), // "pal::RectHandle"
QT_MOC_LITERAL(1, 16, 5), // "moved"
QT_MOC_LITERAL(2, 22, 0), // ""
QT_MOC_LITERAL(3, 23, 3), // "pos"
QT_MOC_LITERAL(4, 27, 11), // "setPosition"
QT_MOC_LITERAL(5, 39, 14), // "setMinPosition"
QT_MOC_LITERAL(6, 54, 14), // "setMaxPosition"
QT_MOC_LITERAL(7, 69, 10), // "setExtents"
QT_MOC_LITERAL(8, 80, 3), // "min"
QT_MOC_LITERAL(9, 84, 3) // "max"

    },
    "pal::RectHandle\0moved\0\0pos\0setPosition\0"
    "setMinPosition\0setMaxPosition\0setExtents\0"
    "min\0max"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_pal__RectHandle[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       5,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,   39,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       4,    1,   42,    2, 0x0a /* Public */,
       5,    1,   45,    2, 0x0a /* Public */,
       6,    1,   48,    2, 0x0a /* Public */,
       7,    2,   51,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void, QMetaType::Double,    3,

 // slots: parameters
    QMetaType::Void, QMetaType::Double,    2,
    QMetaType::Void, QMetaType::Double,    2,
    QMetaType::Void, QMetaType::Double,    2,
    QMetaType::Void, QMetaType::Double, QMetaType::Double,    8,    9,

       0        // eod
};

void pal::RectHandle::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<RectHandle *>(_o);
        (void)_t;
        switch (_id) {
        case 0: _t->moved((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 1: _t->setPosition((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 2: _t->setMinPosition((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 3: _t->setMaxPosition((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 4: _t->setExtents((*reinterpret_cast< double(*)>(_a[1])),(*reinterpret_cast< double(*)>(_a[2]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (RectHandle::*)(double );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&RectHandle::moved)) {
                *result = 0;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject pal::RectHandle::staticMetaObject = { {
    QMetaObject::SuperData::link<QObject::staticMetaObject>(),
    qt_meta_stringdata_pal__RectHandle.data,
    qt_meta_data_pal__RectHandle,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *pal::RectHandle::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *pal::RectHandle::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_pal__RectHandle.stringdata0))
        return static_cast<void*>(this);
    if (!strcmp(_clname, "QGraphicsRectItem"))
        return static_cast< QGraphicsRectItem*>(this);
    return QObject::qt_metacast(_clname);
}

int pal::RectHandle::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
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
void pal::RectHandle::moved(double _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
