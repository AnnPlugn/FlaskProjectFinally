from flask import render_template, redirect, request, url_for, session
import datetime
import tensorflow
from tensorflow.python.keras.layers import LSTM, Dense
from tensorflow.python.keras.models import Sequential
from werkzeug.security import check_password_hash, generate_password_hash
from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:root@db/flaskDB'
app.config['SECRET_KEY'] = ("7dY5syEJUMA6zT8eEMWUA6g6hwyEJUMWUwe88yEJ80mM9Yi39KrY4yA6e880mM9Yi39K"
                            "rYY57XezpsCRIBLYCwtiryEJUA6EYNfdJob1kxDThmJv5Wb6zSFBiHAcSnEcVPmj5d1KK")
db = SQLAlchemy(app)


class POSTS(db.Model):
    __tablename__ = 'Posts'
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), unique=False, nullable=False)
    datetime = db.Column(db.String(20), unique=False, nullable=False)
    author = db.Column(db.String(200), unique=False, nullable=False)
    text = db.Column(db.String(3000), unique=False, nullable=False)


class USERS(db.Model):
    __tablename__ = 'Users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(200), unique=True, nullable=False)
    email = db.Column(db.String(250), unique=True, nullable=False)
    password = db.Column(db.String(1000), unique=False, nullable=False)
    repeat_password = db.Column(db.String(1000), unique=False, nullable=False)
    admin = db.Column(db.Integer, unique=False, nullable=False, default=0)


class DATASET(db.Model):
    __tablename__ = 'Dataset'
    id = db.Column(db.Integer, primary_key=True)
    baths = db.Column(db.String(100), unique=False, nullable=False)
    bedrooms = db.Column(db.String(100), unique=False, nullable=False)
    area = db.Column(db.String(100), unique=False, nullable=False)
    price = db.Column(db.String(100), unique=False, nullable=False)
    predict_price = db.Column(db.String(100), unique=False, nullable=False)
    error = db.Column(db.String(100), unique=False, nullable=False)

class DATASET1(db.Model):
    __tablename__ = 'Dataset1'
    id = db.Column(db.Integer, primary_key=True)
    baths = db.Column(db.String(100), unique=False, nullable=False)
    bedrooms = db.Column(db.String(100), unique=False, nullable=False)
    area = db.Column(db.String(100), unique=False, nullable=False)
    price = db.Column(db.String(100), unique=False, nullable=False)
    predict_price = db.Column(db.String(100), unique=False, nullable=False)
    error = db.Column(db.String(100), unique=False, nullable=False)

with app.app_context():
    db.create_all()
    admins = USERS.query.filter_by(email='a.plugun2005@gmail.com').first()
    if admins is None:
        db.session.add(
            USERS(username='annplugun', email='a.plugun2005@gmail.com',
                  password=generate_password_hash('Annplugun'),
                  repeat_password=generate_password_hash('Annplugun'),
                  admin=1
                  )
        )
        db.session.commit()


async def status(username):
    user = USERS.query.filter_by(username=username).first()
    if user is not None:
        return user.admin
    session.pop('username', None)
    return 0


@app.route('/', methods=['GET', 'POST'])
async def main_fun():
    page = request.args.get('page', 1, type=int)
    posts = POSTS.query.paginate(page=page, per_page=5)
    return render_template("main_view.html",
                           status=await status(session.get('username')),
                           username=session.get('username'),
                           posts=posts)


@app.route('/aboutblog/', methods=['GET', 'POST'])
async def aboutblog_fun():
    return render_template("about_view.html",
                           status=await status(session.get('username')),
                           username=session.get('username'),
                           )


@app.route('/aboutpost/<int:post_id>', methods=['GET', 'POST'])
async def aboutpost_fun(post_id):
    post = POSTS.query.filter_by(id=post_id).first()
    if post is not None:
        return render_template("aboutpost_view.html",
                               status=await status(session.get('username')),
                               username=session.get('username'),
                               post=post)
    return redirect(url_for('main_fun'))


@app.route('/search/', methods=['POST', 'GET'])
async def search():
    search = f"%{request.args.get('search')}%"
    page = request.args.get('page', 1, type=int)
    posts = POSTS.query.filter(
        POSTS.title.like(search) | POSTS.author.like(search) |
        POSTS.text.like(search) | POSTS.datetime.like(search)).paginate(page=page, per_page=5)
    return render_template("main_view.html",
                           status=await status(session.get('username')),
                           username=session.get('username'),
                           posts=posts)


@app.route('/createpost/', methods=['GET', 'POST'])
async def createpost_fun():
    title = request.form.get('title')
    text = request.form.get('text')
    username = session.get('username')
    date = str(datetime.datetime.now().date())
    if title is not None and text is not None and username is not None:
        db.session.add(POSTS(title=title, text=text, author=username, datetime=date))
        db.session.commit()
    return redirect(url_for('main_fun'))


@app.route('/login/', methods=['GET', 'POST'])
async def login_fun():
    email = request.form.get('email')
    password = request.form.get('password')
    if email is not None and password is not None:
        user = USERS.query.filter_by(email=email).first()
        if user is not None:
            if check_password_hash(user.password, password):
                session['username'] = user.username
    return redirect(url_for('main_fun'))


@app.route('/registration/', methods=['GET', 'POST'])
async def registration_fun():
    username = request.form.get('username')
    email = request.form.get('email')
    password1 = request.form.get('password1')
    password2 = request.form.get('password2')
    if username is not None and email is not None and password1 is not None and password2 is not None:
        if password1 == password2:
            ps = generate_password_hash(password1)
            db.session.add(USERS(username=username,
                                 email=email,
                                 password=ps,
                                 repeat_password=ps,
                                 )
                           )
            db.session.commit()
            session['username'] = username
    return redirect(url_for('main_fun'))


@app.route('/logout/')
async def logout_fun():
    session.pop('username', None)
    return redirect(url_for('main_fun'))


# _______________________
@app.route('/polreg/', methods=['GET', 'POST'])
async def polreg_fun():
    return render_template('polreg_view.html',
                           status=await status(session.get('username')),
                           username=session.get('username'),
                           data=DATASET.query.all()
                           )


@app.route('/polreg/start/', methods=['GET', 'POST'])
async def polreg_start_fun():
    DATASET.query.delete()
    data = pd.read_csv('FlaskProjectFirst/DOCKER/DATASET.csv', index_col=0)
    X = data.drop('price', axis=1)[:10000]
    Y = data['price'][:10000]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.01, random_state=42)

    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(X_train.values)

    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, y_train.values)

    X_new_poly = poly_features.transform(X_test.values)
    y_new = lin_reg.predict(X_new_poly)

    for i in range(len(y_new)):
        db.session.add(
            DATASET(baths=str(X_test.values[i][0]),
                    bedrooms=str(X_test.values[i][1]),
                    area=str(X_test.values[i][2]),
                    price=str(int(y_test.values[i])),
                    predict_price=str(int(y_new[i])),
                    error=str(int(y_test.values[i]) - int(y_new[i])),
                    ))
    db.session.commit()
    return redirect(url_for('polreg_fun'))

@app.route('/gbm/', methods=['GET', 'POST'])
async def gbm_fun():
    return render_template('gbm_view.html',
                           status=await status(session.get('username')),
                           username=session.get('username'),
                           data=DATASET.query.all()
                           )

@app.route('/gbm/start1/', methods=['GET', 'POST'])
def gbm_start_fun():
    DATASET.query.delete()
    data = pd.read_csv('FlaskProjectFirst/DOCKER/DATASET.csv', index_col=0)
    X = data.drop('price', axis=1)[:10000]
    Y = data['price'][:10000]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.01, random_state=42)

    gbm = GradientBoostingRegressor()
    gbm.fit(X_train, y_train)

    y_pred = gbm.predict(X_test)

    for i in range(len(y_pred)):
        db.session.add(
            DATASET(baths=str(X_test.values[i][0]),
                     bedrooms=str(X_test.values[i][1]),
                     area=str(X_test.values[i][2]),
                     price=str(int(y_test.values[i])),
                     predict_price=str(int(y_pred[i])),
                     error=str(int(y_test.values[i]) - int(y_pred[i]))
            )
        )
    db.session.commit()
    return redirect(url_for('gbm_fun'))
# _______________________

@app.route('/neyronka/', methods=['GET', 'POST'])
async def neyronka_fun():
    return render_template('neyronka_view.html',
                           status=await statuсвs(session.get('username')),
                           username=session.get('username'),
                           data=DATASET.query.all()
                           )


@app.route('/neyronka/start2/', methods=['GET', 'POST'])
async def neyronka_start_fun():
    DATASET.query.delete()
    data = pd.read_csv('FlaskProjectFirst/DOCKER/DATASET.csv', index_col=0)
    target = np.random.random((100, 1))

    # Подготовка данных
    X = data.drop('price', axis=1)[:10000]
    Y = data['price'][:10000]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.01, random_state=42)

    # Создание модели рекуррентной нейронной сети
    model = Sequential()
    model.add(LSTM(50, input_shape=(X.shape[1],)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Обучение модели
    model.fit(X_train, y_train, epochs=10, batch_size=1)

    # Пример использования обученной модели для предсказания
    new_data_point = np.array([[0.1]])
    new_data_point = new_data_point.reshape((1, 1, 1))
    prediction = model.predict(new_data_point)

    y_new = model.predict(X_test)

    for i in range(len(y_new)):
        db.session.add(
            DATASET(baths=str(X_test.values[i][0]),
                    bedrooms=str(X_test.values[i][1]),
                    area=str(X_test.values[i][2]),
                    price=str(int(y_test.values[i])),
                    predict_price=str(int(y_new[i])),
                    error=str(int(y_test.values[i]) - int(y_new[i]))
                    )
        )
    db.session.commit()

    return redirect(url_for('neyronka_fun'))
@app.route('/adminpanel/', methods=['POST', 'GET'])
async def adminpanel_fun():
    return render_template('adminpanel_view.html',
                           status=await status(session.get('username')),
                           username=session.get('username'),
                           )


@app.route('/adminpanel/posts/', methods=['POST', 'GET'])
async def adminpanel_posts_fun():
    page = request.args.get('page', 1, type=int)
    posts = POSTS.query.paginate(page=page, per_page=5)
    users = USERS.query.all()
    return render_template('adminpanel_posts_view.html',
                           status=await status(session.get('username')),
                           username=session.get('username'),
                           posts=posts,
                           authors=[user.username for user in users]
                           )


@app.route('/adminpanel/users/', methods=['POST', 'GET'])
async def adminpanel_users_fun():
    page = request.args.get('page', 1, type=int)
    users = USERS.query.paginate(page=page, per_page=5)
    return render_template('adminpanel_users_view.html',
                           status=await status(session.get('username')),
                           username=session.get('username'),
                           users=users,
                           )


@app.route('/adminpanel/operations/changetitle/<int:post_id>/', methods=['POST', 'GET'])
async def adminpanel_operations_changetitle_fun(post_id):
    input = request.args.get('input')
    if input is not None:
        post = POSTS.query.filter_by(id=post_id).first()
        if post is not None:
            post.title = input
            db.session.commit()
    return redirect(url_for('adminpanel_posts_fun'))


@app.route('/adminpanel/operations/changeauthor/<int:post_id>/', methods=['POST', 'GET'])
async def adminpanel_operations_changeauthor_fun(post_id):
    input = request.args.get('input')
    if input is not None:
        post = POSTS.query.filter_by(id=post_id).first()
        if post is not None:
            post.author = input
            db.session.commit()
    return redirect(url_for('adminpanel_posts_fun'))


@app.route('/adminpanel/operations/changetext/<int:post_id>/', methods=['POST', 'GET'])
async def adminpanel_operations_changetext_fun(post_id):
    input = request.args.get('input')
    if input is not None:
        post = POSTS.query.filter_by(id=post_id).first()
        if post is not None:
            post.text = input
            db.session.commit()
    return redirect(url_for('adminpanel_posts_fun'))


@app.route('/adminpanel/operations/deletepost/<int:post_id>/', methods=['POST', 'GET'])
async def adminpanel_operations_deletepost_fun(post_id):
    db.session.delete(POSTS.query.filter_by(id=post_id).first())
    db.session.commit()
    return redirect(url_for('adminpanel_posts_fun'))


@app.route('/adminpanel/operations/changeusername/<int:user_id>/', methods=['POST', 'GET'])
async def adminpanel_operations_changeusername_fun(user_id):
    input = request.args.get('input')
    if input is not None:
        user = USERS.query.filter_by(id=user_id).first()
        if user is not None:
            user.username = input
            db.session.commit()
    return redirect(url_for('adminpanel_users_fun'))


@app.route('/adminpanel/operations/deleteuser/<int:user_id>/', methods=['POST', 'GET'])
async def adminpanel_operations_deleteuser_fun(user_id):
    db.session.delete(USERS.query.filter_by(id=user_id).first())
    db.session.commit()
    return redirect(url_for('adminpanel_users_fun'))


if __name__ == '__main__':
    app.run(host="0.0.0.0")
