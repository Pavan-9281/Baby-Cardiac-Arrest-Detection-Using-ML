# Generated manually to add missing models

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Remote_User', '0007_clientposts_model_names'),
    ]

    operations = [
        migrations.CreateModel(
            name='cardiac_arrest_prediction',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Fid', models.CharField(max_length=3000)),
                ('Age_In_Days', models.CharField(max_length=3000)),
                ('Sex', models.CharField(max_length=3000)),
                ('ChestPainType', models.CharField(max_length=3000)),
                ('RestingBP', models.CharField(max_length=3000)),
                ('RestingECG', models.CharField(max_length=3000)),
                ('MaxHR', models.CharField(max_length=3000)),
                ('ExerciseAngina', models.CharField(max_length=3000)),
                ('Oldpeak', models.CharField(max_length=3000)),
                ('ST_Slope', models.CharField(max_length=3000)),
                ('slp', models.CharField(max_length=3000)),
                ('caa', models.CharField(max_length=3000)),
                ('thall', models.CharField(max_length=3000)),
                ('Prediction', models.CharField(max_length=3000)),
            ],
        ),
        migrations.CreateModel(
            name='detection_accuracy',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('names', models.CharField(max_length=300)),
                ('ratio', models.CharField(max_length=300)),
            ],
        ),
        migrations.CreateModel(
            name='detection_ratio',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('names', models.CharField(max_length=300)),
                ('ratio', models.CharField(max_length=300)),
            ],
        ),
    ] 