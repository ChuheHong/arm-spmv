#ifndef VECTOR_H
#define VECTOR_H

class Vector
{
public:
    int     size;
    double* values;

    Vector();
    Vector(int n, double* values);
    Vector(const Vector& x);
    ~Vector();
    Vector& operator=(double a);
    Vector& operator=(const Vector& x);

    void Free();
    void Resize(int n);
    void Fill(double a) const;
    void FillRandom() const;
    void Copy(const Vector& x) const;
    void Scale(double a) const;
    void Shift(double a) const;
    void AddScaled(double a, const Vector& x) const;
    void Add2Scaled(double a, const Vector& x, double b, const Vector& y) const;
};

#endif  // VECTOR_H